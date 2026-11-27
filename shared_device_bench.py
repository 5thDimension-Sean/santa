"""Paired, location-guided CPU/OpenCL search, with full-program replay."""
from collections import deque
import csv
import ctypes
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pyopencl as cl
from near_reductions import rank, unfocused_rows
from optimize import compile_graph, graph, load_puzzles, read_bank, replay
from elf import write_submission

BASE = Path('experiments')
TICK = time.perf_counter
BUDGET = 1.0


class Engine:
    def __init__(self, gpu):
        self.gpu = gpu
        if gpu:
            device = next(d for p in cl.get_platforms() for d in p.get_devices()
                          if 'GTX 1650 SUPER' in d.name)
            self.ctx = cl.Context([device])
            self.queue = cl.CommandQueue(self.ctx)
            program = cl.Program(self.ctx, Path('shared_bench.cl').read_text()).build()
            self.kernels = {k.function_name:k for k in program.all_kernels()}
        else:
            self.runtime = os.add_dll_directory(r'C:\msys64\ucrt64\bin')
            self.dll = ctypes.CDLL(str(Path(tempfile.gettempdir())/'shared_bench_near_cpu.dll'))

    def run(self, name, count, *args, out):
        if self.gpu:
            flags = cl.mem_flags
            inputs = [cl.Buffer(self.ctx, flags.READ_ONLY|flags.COPY_HOST_PTR, hostbuf=x)
                      if isinstance(x,np.ndarray) else x for x in args]
            dest = cl.Buffer(self.ctx,flags.WRITE_ONLY,out.nbytes)
            self.kernels[name](self.queue,(count,),None,*inputs,dest)
            cl.enqueue_copy(self.queue,out,dest,is_blocking=True)
        else:
            args = [ctypes.c_void_p(x.ctypes.data) if isinstance(x,np.ndarray)
                    else ctypes.c_int64(int(x)) if isinstance(x,np.int64)
                    else ctypes.c_int(int(x)) for x in args]
            getattr(self.dll,name)(*args,ctypes.c_void_p(out.ctypes.data))
        return out


def visit(targets, nodes, stop=()):
    seen = set(stop)
    order = []
    def dfs(v):
        if v in seen or v not in nodes: return
        seen.add(v)
        for x in nodes[v][1:]: dfs(x)
        order.append(v)
    for t in targets: dfs(t)
    return order


def regions(p, nodes, locations, rng):
    pool = sorted(locations.intersection(nodes)) or list(nodes)
    rng.shuffle(pool)
    deps = {v:set(visit([v],nodes)) for v in pool}
    # Try single outputs and pairs with overlapping dependencies.
    for v in pool:
        yield (v,)
        others = [x for x in pool if x!=v and x not in deps[v] and v not in deps[x]]
        rng.shuffle(others)
        others.sort(key=lambda x: len(deps[x]&deps[v]) / len(deps[x]|deps[v]),reverse=True)
        if others: yield (v,others[0])
        if len(others)>1: yield (v,rng.choice(others[1:]))


def operands(t, x, limit):
    # Each expression evaluates to t when the placeholder is replaced by v.
    options = [(t-x, '+', False), (t+x, '-', False), (x-t, '-', True)]
    if t%x == 0: options.append((t//x, '*', False))
    if t<=limit//x: options.append((t*x, '/', False))
    if x%t == 0: options.append((x//t, '/', True))
    for v,op,reverse in options:
        if 0<v<=limit and v!=t:
            yield v, (op,x,v) if reverse else (op,v,x)


def sample_suffixes(t, free, limit, rng):
    # A residual root can expose a deeper expression ending in one or two
    # operations with retained values. Always include the unpeeled root.
    yield ()
    values=sorted(free)
    first=[]
    for x in rng.sample(values,min(4,len(values))):
        first.extend(((t,v,recipe),) for v,recipe in operands(t,x,limit) if v not in free)
    rng.shuffle(first)
    for tail in first[:3]:
        yield tail
        v=tail[-1][1]
        x=rng.choice(values)
        second=[(v,w,recipe) for w,recipe in operands(v,x,limit)
                if w not in free and w!=t]
        if second: yield tail+(rng.choice(second),)


def repairs(engine,p,prog,nodes,roots,end,stats,rng,forced_suffix=None, focus=(), observe=None, use_reduction=False, collect_patch=None):
    keep = visit(p.targets,nodes,roots)
    available = set(p.starts)
    for v in keep:
        if all(x in available for x in nodes[v][1:]): available.add(v)
    forbidden = set(keep).difference(available).union(roots)
    allowance = len(prog)-len(keep)-len(roots)
    if allowance<0 and collect_patch is None: return
    usable = set(focus) & available
    if focus and not usable: return
    live = np.array(sorted(available),dtype=np.int64)
    n = len(live)
    raw = engine.run('make_values',n*n,live,np.int32(n),np.int64(p.limit),
                     out=np.empty(4*n*n,dtype=np.int64))
    vals,indices = np.unique(np.concatenate((live,raw)),return_index=True)
    mask = (vals>0) & ~np.isin(vals,list(forbidden))
    vals,indices = vals[mask],indices[mask]
    cost = (indices>=n).astype(np.int32)
    def recipe(pos):
        pair,op = divmod(int(indices[pos])-n,4)
        i,j = divmod(pair,n)
        a,b = int(live[i]),int(live[j])
        if op in (1,3): a,b=b,a
        return ('+-*/'[op],a,b)
    near = bool((focus or use_reduction) and len(roots)==1 and forced_suffix is None)
    helper_rows=np.arange(len(vals),dtype=np.int32)
    if focus:
        fresh_input=np.isin(live,list(usable))
        chosen=np.maximum(indices-n,0)//4
        focused=np.isin(vals,list(usable)) | ((indices>=n) &
                (fresh_input[chosen//n] | fresh_input[chosen%n]))
        helper_rows=np.flatnonzero(focused)
    if near and not focus:
        helper_rows=unfocused_rows(cost)
    if near:
        t=roots[0]
        ranked=rank(engine,t,live,vals,cost,helper_rows,p.limit,forbidden)
        suffixes=[()]
        for position,(_,v,k,r,_,price) in enumerate(ranked[:64],1):
            if price==127: continue
            suffixes.append(((t,v,r),))
            if observe is not None: observe('backward proposal',(v,k),roots,position)
    elif forced_suffix is not None:
        suffixes=[forced_suffix]
    elif len(roots)==1:
        suffixes=list(sample_suffixes(roots[0],available,p.limit,rng))
    else:
        suffixes=[()]
    probes={}
    for suffix in suffixes:
        if len(suffix)>allowance and collect_patch is None: continue
        for t in roots:
            v=suffix[-1][1] if suffix else t
            if not suffix or v not in forbidden:
                probes.setdefault(v,[]).append(suffix)
    if not probes: return
    targets=list(probes)
    m=len(targets)
    helpers=vals[helper_rows]
    out=engine.run('frontier_links',len(helpers)*m,vals,cost,np.int32(len(vals)),
                   helpers,np.int32(len(helpers)),np.array(targets,dtype=np.int64),
                   np.int32(m),np.int64(p.limit),out=np.empty((len(helpers),m),dtype=np.int32))
    stats[0]+=1
    stats[1]+=len(helpers)*m
    if focus and observe is not None and TICK()<end:
        observe('tested',usable,roots,len(helpers)*m)
    if len(roots)>1:
        matches=((int(pos),list(range(m)),()) for pos in np.flatnonzero((out>=0).all(axis=1)))
    else:
        matches=((int(pos),[int(col)],suffix) for pos,col in zip(*np.nonzero(out>=0))
                 for suffix in probes[targets[col]])
    for pos,cols,suffix in matches:
        if TICK()>=end: return
        patch = {}
        row=pos
        pos=int(helper_rows[row])
        h = int(vals[pos])
        if cost[pos]: patch[h]=recipe(pos)
        for col in cols:
            t=targets[col]
            qi,op = divmod(int(out[row,col]),6)
            x = int(vals[qi])
            if cost[qi]: patch[x]=recipe(qi)
            a,b = (x,h) if op in (2,5) else (h,x)
            patch[t]=('+--*//'[op],a,b)
        # Disallow a suffix that would introduce a cycle via one of its helpers.
        if any(v in patch for v,_,_ in suffix): continue
        for v,w,r in reversed(suffix):
            k=r[2] if r[1]==w else r[1]
            if k not in available:
                i=int(vals.searchsorted(k))
                if i==len(vals) or vals[i]!=k: break
                patch[k]=recipe(i)
            patch[v]=r
        if not all(t in patch for t in roots): continue
        if collect_patch is not None:
            collect_patch(patch)
            continue
        if len(patch)-len(roots)>allowance: continue
        updated = nodes|patch
        order = visit(p.targets,updated)
        if len(order)>len(prog): continue
        if len(order)==len(prog) and all(v in nodes for v in order): continue
        # A residual that uses itself as an input is not an executable recipe.
        if any(v in r[1:] for v,r in patch.items()): continue
        candidate = compile_graph(p,updated,order)
        if candidate is None: continue
        replay(p,candidate)
        if TICK()<end: yield candidate


def followup_outputs(p,nodes,locations,fresh):
    # Prefer targets whose current calculation has the most removable work.
    # Rank ties in puzzle order. A helper must survive the cut to be reusable.
    ranked=[]
    for t in p.targets:
        if t not in nodes or (locations and t not in locations): continue
        keep=visit(p.targets,nodes,(t,))
        if fresh.isdisjoint(keep): continue
        removable=len(nodes)-len(keep)
        if removable>=2: ranked.append((removable,t))
    ranked.sort(key=lambda item:-item[0])
    return [t for _,t in ranked[:4]]


def solve(engine,p,old,locations,observe=None):
    start = TICK()
    end = start+BUDGET
    best=old
    stats=[0,0]
    rng=random.Random(781+int(p.id[-4:]))
    frontier=[old]
    seen={frozenset(graph(p,old))}
    followed=set()
    def remember(candidate,signature):
        nonlocal best,frontier
        seen.add(signature)
        if len(candidate)<len(best):
            best=candidate
            frontier=[best]
        elif len(candidate)==len(best):
            if len(frontier)>=12: frontier.pop(rng.randrange(1,len(frontier)))
            frontier.append(candidate)
    while TICK()<end:
        prog=rng.choice(frontier)
        nodes=graph(p,prog)
        for count,roots in enumerate(regions(p,nodes,locations,rng)):
            if TICK()>=end or count>=12: break
            for candidate in repairs(engine,p,prog,nodes,roots,end,stats,rng,
                                     use_reduction=stats[0]%12==11):
                updated=graph(p,candidate)
                signature=frozenset(updated)
                if signature in seen: continue
                remember(candidate,signature)
                pending=deque()
                def enqueue(program,graph_after,graph_before,source):
                    fresh=frozenset(graph_after).difference(graph_before,followed)
                    if not fresh or len(program)>len(best): return
                    followed.update(fresh)
                    if observe is not None: observe('discovered',fresh,source,0)
                    pending.append((program,graph_after,fresh))
                enqueue(candidate,updated,nodes,roots)
                # Process discoveries breadth first, including new values
                # introduced by a follow-up, before random frontier eviction.
                while pending and TICK()<end:
                    state,state_nodes,fresh=pending.popleft()
                    if len(state)>len(best): continue
                    for t in followup_outputs(p,state_nodes,locations,fresh):
                        if TICK()>=end: break
                        for second in repairs(engine,p,state,state_nodes,(t,),end,stats,rng,
                                              focus=fresh,observe=observe):
                            next_nodes=graph(p,second)
                            next_signature=frozenset(next_nodes)
                            if next_signature in seen: continue
                            remember(second,next_signature)
                            if observe is not None:
                                observe('accepted follow-up',fresh,(t,),len(second))
                            enqueue(second,next_nodes,state_nodes,(t,))
    return best,*stats,TICK()-start


def parity(cpu,gpu):
    for live in ([1,4,12,17,40],[1,2,10**9,10**17+3]):
        v=np.array(live,dtype=np.int64)
        n=len(v); limit=np.int64(10**18)
        def check(name,count,args,dtype):
            x,y=[e.run(name,count,*args,out=np.empty(count*(4 if name=='make_values' else 1),dtype=dtype)) for e in (cpu,gpu)]
            assert np.array_equal(x,y)
            return x
        raw=check('make_values',n*n,[v,np.int32(n),limit],np.int64)
        vals=np.unique(np.concatenate((v,raw))); vals=vals[vals>0]
        roots=np.array([48,60,357],dtype=np.int64)
        check('links',len(vals)*3,[vals,(~np.isin(vals,v)).astype(np.int32),np.int32(len(vals)),roots,np.int32(3),limit],np.int32)


def main(trace_puzzle=None):
    puzzles=load_puzzles('puzzles.csv')
    older=read_bank('public/ishankarnick/submission.csv',puzzles)
    current=read_bank('submission.csv',puzzles,strict=True)
    assert sum(map(len,current.values()))==127311
    keys=[k for k in older if len(older[k])>len(current[k])]
    assert len(keys)==91 and sum(len(older[k])-len(current[k]) for k in keys)==130
    cpu,gpu=Engine(False),Engine(True)
    parity(cpu,gpu)
    names=['cpu','gpu']; devices=dict(zip(names,[cpu,gpu]))
    totals=dict.fromkeys(names,0)
    recovered=dict.fromkeys(names,0)
    with (BASE/'shared-device-reduction-benchmark.csv').open('w',newline='') as f:
        out=csv.writer(f)
        out.writerow(['id','device','known_savings','old_steps','verified_steps','saved','recovered','budget_seconds','elapsed_seconds','regions','helper_root_tests'])
        with (BASE/'shared-device-reduction-hits.jsonl').open('w') as hits:
            for i,k in enumerate(keys):
                p=puzzles[k]
                a,b=graph(p,older[k]),graph(p,current[k])
                locations={v for v in a if a[v]!=b.get(v)}
                for name in (names if i%2==0 else names[::-1]):
                    events=[]
                    def record(event,values,roots,count):
                        events.append((event,sorted(values),list(roots),count))
                    prog,n,tests,seconds=solve(devices[name],p,older[k],locations,
                                               record if k==trace_puzzle else None)
                    if k==trace_puzzle:
                        path=BASE/f'shared-device-reduction-trace-{k}-{name}.json'
                        path.write_text(json.dumps(events))
                    length=replay(p,prog)
                    delta=len(older[k])-length
                    known=len(older[k])-len(current[k])
                    totals[name]+=delta
                    recovered[name]+=min(known,delta)
                    out.writerow([k,name,known,len(older[k]),length,delta,min(known,delta),BUDGET,seconds,n,tests])
                    if delta:
                        print(json.dumps(dict(id=k,device=name,program=prog)),file=hits,flush=True)
                f.flush()
                if (i+1)%15==0: print(i+1,'of',len(keys),totals,flush=True)
    print('verified historical gains',totals,'capped recovery',recovered,flush=True)
    if totals['gpu']>totals['cpu']:
        best=dict(current)
        with (BASE/'shared-device-baseline.csv').open('w',newline='') as f:
            out=csv.writer(f); out.writerow(['id','before','after','saved','regions','helper_root_tests','elapsed_seconds'])
            for i,(k,p) in enumerate(puzzles.items()):
                prog,n,tests,seconds=solve(gpu,p,current[k],set())
                best[k]=prog; gain=len(current[k])-replay(p,prog)
                out.writerow([k,len(current[k]),len(prog),gain,n,tests,seconds])
                if (i+1)%50==0:
                    f.flush()
                    print('baseline',i+1,'of 1000, saved',127311-sum(map(len,best.values())),flush=True)
        path=BASE/'shared-device-submission.csv'
        write_submission(path,puzzles,best)
        score=sum(map(len,read_bank(path,puzzles,strict=True).values()))
        print('final verified score',score,'saved',127311-score,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace-puzzle',help='Record selection events for this puzzle')
    main(parser.parse_args().trace_puzzle)

