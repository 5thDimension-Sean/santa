"""Select shared arithmetic recipes for a group of targets with a local MILP."""
import argparse
from collections import Counter, defaultdict
import json
from math import gcd
import os
from pathlib import Path
import random
import subprocess
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from optimize import SOURCES, compile_graph, graph, load_puzzles, read_bank, replay

metrics=Counter()

def ancestors(outputs, graph, free=()):
    found=set()
    fixed=set(free)
    stack=list(outputs)
    while stack:
        v=stack.pop()
        if v not in graph or v in found or v in fixed: continue
        found.add(v)
        stack.extend(graph[v][1:])
    return found

def enumerate_pairs(values,limit):
    vs=sorted(values)
    for i,a in enumerate(vs):
        for b in vs[:i+1]:
            if a+b<=limit: yield a+b,('+',a,b)
            if a>b: yield a-b,('-',a,b)
            if a*b<=limit: yield a*b,('*',a,b)
            if a%b==0: yield a//b,('/',a,b)

def shared_bases(roots,free,direct,limit,rng):
    # Infer h from several equations t = k*h +/- e. Offsets and multipliers
    # must themselves be available or have a cheap construction.
    multipliers=sorted({k for k in free|set(range(2,65)) if 1<k<=4096 and (k in free or k in direct)})
    offsets=[0,*free]
    hits=defaultdict(dict)
    for t in roots:
        for e in offsets:
            for product in {t-e,t+e}:
                if not 0<product<=limit: continue
                for k in multipliers:
                    h,rem=divmod(product,k)
                    if rem or h<2 or h in free or h in roots: continue
                    hits[h].setdefault(t,(k,product))
    shared=[h for h,targets in hits.items() if len(targets)>1]
    rng.shuffle(shared)
    shared.sort(key=lambda h:(h not in direct,-len(hits[h])))
    extra=set()
    for h in shared[:32]:
        operands={h}
        if h not in direct:
            matches=[]
            for f in free:
                for x in (h-f,h+f,f-h,h//f if h%f==0 else 0,h*f,f//h if f%h==0 else 0):
                    if x in direct: matches.append(x)
            if not matches: continue
            operands.update(matches[:8])
        for k,product in hits[h].values(): operands.update((k,product))
        extra.update(operands-free)
    return extra

def stage_outputs(p,old,rng,kind):
    level=dict.fromkeys(p.starts,0)
    blocks=defaultdict(set)
    users=defaultdict(set)
    for v,(op,x,y) in old.items():
        participating=op in kind
        level[v]=max(level[x],level[y])+int(not participating)
        if participating: blocks[level[v]].add(v)
        users[x].add(v);users[y].add(v)
    groups=[]
    for block in blocks.values():
        boundary=[v for v in block if v in p.targets or users[v]-block]
        if len(block)>=4 and len(boundary)>=2: groups.append((block,boundary))
    if not groups: return set()
    block,boundary=rng.choices(groups,weights=[len(b) for b,_ in groups])[0]
    if len(boundary)<=12: return set(boundary)
    pivot=rng.choice(sorted(boundary))
    return set(sorted(boundary,key=lambda v:gcd(pivot,v).bit_length()+5*rng.random(),reverse=True)[:12])

def divisor_pool(roots,free,direct,rng):
    ports=sorted(roots)
    run=subprocess.run([Path('experiments/number-recipes.exe').resolve(),'--divisors'],
                       input=' '.join(map(str,ports)),text=True,capture_output=True,check=True,timeout=60)
    lists=[set(map(int,line.split())) for line in run.stdout.splitlines()]
    count=Counter(d for divs in lists for d in divs if d not in free)
    pool=set()
    for v,divs in zip(ports,lists):
        pairs=[(d,v//d) for d in divs if 1<d<=v//d]
        def rank(pair):
            d,e=pair
            return (int(d in free)+int(e in free),int(d in direct)+int(e in direct),
                    count[d]+count[e],rng.random())
        for d,e in sorted(pairs,key=rank,reverse=True)[:12]: pool.update((d,e))
        # Include small factors and prime powers so novel composite factors
        # can be reached even when none was in the initial arithmetic graph.
        small=sorted(d for d in divs if 1<d<=128)
        pool.update(small[:16])
    return pool-free

def linear_pool(ports,old,free,limit,rng):
    # Reassociate several signed sums at once. All arithmetic here describes
    # candidate values; only legal register operations can enter the solution.
    vectors={v:Counter({v:1}) for v in free}
    for v,(op,x,y) in old.items():
        if v in vectors: continue
        if op not in ('+','-') or x not in vectors or y not in vectors:
            vectors[v]=Counter({v:1})
        else:
            c=vectors[x].copy()
            c.update(vectors[y]) if op=='+' else c.subtract(vectors[y])
            vectors[v]=Counter({w:k for w,k in c.items() if k})
    counts=Counter()
    for port in ports:
        terms=[x*k for x,k in vectors[port].items()]
        if not 2<=len(terms)<=16: continue
        used=set()
        for _ in range(4 if len(terms)>9 else 1):
            chosen=terms if len(terms)<=9 else rng.sample(terms,9)
            sums={0}
            for x in chosen: sums.update([s+x for s in sums])
            for s in sums:
                w=abs(s)
                if 0<w<=limit and w not in free: used.add(w)
        counts.update(used)
        for x,k in vectors[port].items():
            if abs(k)>1 and abs(k*x)<=limit: counts[abs(k*x)]+=1
    candidates=list(counts)
    rng.shuffle(candidates);candidates.sort(key=counts.get,reverse=True)
    return set(candidates[:190])

def proposals(p,base,donors,rng,group=None,backbone=False,factors=False,products=False,linear=False):
    old=graph(p,base)
    targetset=set(p.targets)
    leaves=targetset.intersection(old)-{a for r in old.values() for a in r[1:]}
    if products or linear:
        group=stage_outputs(p,old,rng,'*/' if products else '+-')
        if not group: return
    if backbone and group is None: group=set()
    if group is None and len(leaves)<2: return
    if group is None:
        chains={t:ancestors([t],old) for t in leaves}
        root=rng.choice(sorted(leaves))
        similarity={t:len(chains[t]&chains[root])/len(chains[t]|chains[root])+0.55*rng.random() for t in leaves}
        group={root,*sorted(leaves,key=similarity.get,reverse=True)[:rng.randint(3,8)]}
    if backbone:
        counts=Counter(x for r in old.values() for x in set(r[1:]))
        helpers=[v for v in old if counts[v]>=2 and v not in p.starts]
        if len(helpers)<3: return
        pivot=rng.choice(helpers)
        group={pivot}
        vicinity=[v for v,r in old.items() if pivot in r[1:] and v not in targetset]
        if vicinity:
            group.update(rng.sample(vicinity,min(len(vicinity),rng.randint(1,3))))
        group.update(rng.sample(helpers,min(len(helpers),rng.randint(2,5))))
    core=ancestors(targetset,old,group)
    roots=group & (targetset|{a for v in core for a in old[v][1:]})
    free=set(p.starts)
    for v,r in old.items():
        if v in core and r[1] in free and r[2] in free: free.add(v)
    forbidden=core-free
    replace=set(old)-core
    if not roots or len(replace)<4:
        metrics['small or empty cut']+=1
        return
    pool=replace.copy()
    known=defaultdict(set)
    for v in replace: known[v].add(old[v])
    for donor in donors:
        dg=graph(p,donor)
        branch=ancestors(roots,dg,free)
        if len(branch)>len(replace)+28 or branch&forbidden: continue
        pool.update(branch)
        for v in branch: known[v].add(dg[v])
    # Harvest new values that join several different outputs to the surviving
    # part of the program. Keep one way to construct each in a single step.
    direct={}
    for v,r in enumerate_pairs(free,p.limit):
        if v not in free and v not in forbidden: direct.setdefault(v,r)
    if backbone:
        pool.update(shared_bases(roots,free,direct,p.limit,rng))
    if products:
        pool.update(divisor_pool(roots,free,direct,rng)-forbidden)
    if linear:
        pool.update(linear_pool(roots,old,free,p.limit,rng)-forbidden)
    if factors:
        pivots=sorted(roots)
        extra=list(replace-roots)
        rng.shuffle(extra)
        pivots+=extra[:max(0,10-len(pivots))]
        case=' '.join(map(str,[p.limit,rng.randrange(2**32),len(free),len(pivots),*sorted(free),*pivots]))
        proc=subprocess.run([Path('experiments/number-recipes.exe').resolve()],input=case,
                            text=True,capture_output=True,check=True,timeout=60)
        pool.update(set(map(int,proc.stdout.split()))-free-forbidden)
    for a in group:
        for b in group:
            if a>=b: continue
            d=gcd(a,b)
            pool.update({d,a//d,b//d,b-a}-free)
    links=Counter()
    for v in pool:
        for f in free:
            for q in (v-f,v+f,f-v,v//f if v%f==0 else 0,v*f,f//v if f%v==0 else 0):
                if q in direct and q not in pool: links[q]+=1
    extras=[v for v,n in links.items() if n>=2]
    rng.shuffle(extras)
    extras.sort(key=links.get,reverse=True)
    pool.update(extras[:120])
    # Carry a few alternative factors of cheap target offsets into the model.
    # Even a single-use new value may unlock a simultaneous change elsewhere.
    relevant=list(roots)
    for t in relevant:
        offsets=list(free)
        rng.shuffle(offsets)
        for f in offsets[:24]:
            for v in (t+f,t-f):
                if not 0<v<=p.limit: continue
                for a in (2,3,4,5,7,9,11,13,16,17,19,23,25,29,31,32,37,64,81,125,127,128,256):
                    if v%a or (a not in free and a not in direct): continue
                    b=v//a
                    if b not in free and b not in direct: continue
                    pool.update({v,a,b}-free)
    for v in pool:
        if v in direct: known[v].add(direct[v])
    pool-=forbidden
    for v,r in enumerate_pairs(free|pool,p.limit):
        if v in pool and v not in r[1:]: known[v].add(r)
    # Recipes with identical optional operands have identical costs. Retain
    # the original one on ties to improve register allocation.
    recipes={}
    for v in sorted(pool):
        compact={}
        for r in sorted(known[v],key=lambda r:(r!=old.get(v),r)):
            dep=tuple(sorted(set(r[1:])-free))
            if v not in dep: compact.setdefault(dep,r)
        if () in compact: compact={():compact[()]}
        recipes[v]=compact
    for choices,fixed_order in ordered_models(recipes,rng):
        for selected in select(choices,roots,len(base)-len(core)-1,fixed_order):
            merged={v:old[v] for v in core}
            merged.update(selected)
            needed=ancestors(targetset,merged)
            if len(needed)>=len(base): continue
            for i in range(12):
                program=schedule(p,merged,needed,old,rng,i)
                if program is not None:
                    replay(p,program)
                    yield program
                    return
            metrics['allocation failures']+=1

def ordered_models(recipes,rng):
    # Two inexpensive orientations, then a model that can choose its own
    # instruction order. A mutually supporting cycle cannot pay for a value.
    for use_max in (True,False):
        cost={}
        while True:
            changed=False
            for v,options in recipes.items():
                for dep in options:
                    if not all(a in cost for a in dep): continue
                    depth=1+(max((cost[a] for a in dep),default=0) if use_max else sum(cost[a] for a in dep))
                    if depth<cost.get(v,1e12): cost[v]=depth;changed=True
            if not changed: break
        order={v:(c,rng.random()) for v,c in cost.items()}
        yield {v:{dep:r for dep,r in options.items() if v in order and all(a in order and order[a]<order[v] for a in dep)}
               for v,options in recipes.items()},True
    yield recipes,False

def schedule(p,nodes,needed,original,rng,attempt):
    ranks={v:i for i,v in enumerate(original)}
    users=defaultdict(set)
    indeg={}
    for v in needed:
        parents=set(nodes[v][1:])
        indeg[v]=len(parents&needed)
        for x in parents: users[x].add(v)
    remaining={x:len(d) for x,d in users.items()}
    ready=[v for v in needed if not indeg[v]]
    order=[]
    while ready:
        def priority(v):
            a,b=nodes[v][1:]
            release=(remaining[a]==1)+(a!=b and remaining[b]==1)+(remaining.get(v,0)==0)
            return -release*(5 if attempt==0 else 400)+ranks.get(v,100)+attempt*rng.random()*10
        v=min(ready,key=priority)
        ready.remove(v)
        order.append(v)
        for x in set(nodes[v][1:]): remaining[x]-=1
        for x in users[v]:
            indeg[x]-=1
            if not indeg[x]: ready.append(x)
    return compile_graph(p,nodes,order) if len(order)==len(needed) else None

def select(recipes,roots,budget,fixed_order):
    values=list(recipes)
    vi={v:i for i,v in enumerate(values)}
    costs=[1.]*len(values)
    lower=[float(v in roots) for v in values]
    upper=[1.]*len(values)
    integrality=[1]*len(values)
    if not fixed_order:
        costs.extend([0.]*len(values))
        lower.extend([0.]*len(values))
        upper.extend([float(budget)]*len(values))
        integrality.extend([0]*len(values))
    row,col,data,lo,hi=[],[],[],[],[]
    def constraint(terms,l,h):
        i=len(lo)
        for j,c in terms: row.append(i);col.append(j);data.append(c)
        lo.append(l);hi.append(h)
    edges=[]
    for v,options in recipes.items():
        support=[]
        for dep,recipe in options.items():
            if any(a not in vi for a in dep): continue
            j=len(costs)
            costs.append(0.);integrality.append(int(not fixed_order));lower.append(0.);upper.append(1.)
            support.append((j,1))
            edges.append((v,dep,recipe,j))
            for a in dep:
                constraint([(j,1),(vi[a],-1)],-np.inf,0)
                if not fixed_order:
                    maximum=budget+1
                    constraint([(len(values)+vi[v],1),(len(values)+vi[a],-1),(j,-maximum)],1-maximum,np.inf)
        constraint(support+[(vi[v],-1)],0,0)
    constraint([(vi[v],1) for v in values],-np.inf,budget)
    bounds=Bounds(lower,upper)
    for _ in range(6):
        mat=coo_matrix((data,(row,col)),shape=(len(lo),len(costs))).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected')
            result=milp(costs,integrality=integrality,bounds=bounds,constraints=LinearConstraint(mat,lo,hi),
                        options={'time_limit':1.5,'threads':1,'mip_rel_gap':0.})
        metrics['solver status '+str(result.status)]+=1
        if result.x is None: return
        active={v for v in values if result.x[vi[v]]>.5}
        picked={}
        while True:
            count=len(picked)
            for v,dep,recipe,_ in edges:
                if v in active and v not in picked and all(a in picked for a in dep): picked[v]=recipe
            if count==len(picked): break
        if roots<=picked.keys():
            metrics['valid derivations']+=1
            yield picked
            # Explore a different equally cheap set if allocation fails.
            constraint([(vi[v],1) for v in active],-np.inf,len(active)-1)
        else:
            unreachable=active-picked.keys()
            incoming=[(j,1) for v,dep,_,j in edges if v in unreachable and not unreachable.intersection(dep)]
            if not incoming: return
            constraint(incoming,1,np.inf)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=24398)
    ap.add_argument('--count',type=int,default=1000)
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--backbone',action='store_true')
    ap.add_argument('--factors',action='store_true')
    ap.add_argument('--products',action='store_true')
    ap.add_argument('--linear',action='store_true')
    args=ap.parse_args()
    puzzles=load_puzzles('puzzles.csv')
    best=read_bank('submission.csv',puzzles,strict=True)
    banks=[read_bank(s,puzzles) for s in SOURCES]
    ids=[k for k,p in puzzles.items() if p.limit>16]
    rng=random.Random(args.seed)
    rng.shuffle(ids)
    ids=ids[args.shard::args.shards][:args.count]
    saved=0
    for i,k in enumerate(ids,1):
        seen=set()
        donors=[]
        for bank in banks:
            sig=tuple(map(tuple,bank[k]))
            if sig not in seen: donors.append(bank[k]);seen.add(sig)
        for prog in proposals(puzzles[k],best[k],donors,rng,backbone=args.backbone,factors=args.factors,products=args.products,linear=args.linear):
            saved+=len(best[k])-len(prog)
            best[k]=prog
            dest=Path('experiments')/f'region-{args.seed}-{k}.json'
            temp=dest.with_suffix('.pending')
            temp.write_text(json.dumps({'id':k,'program':prog,'origin':f'local regional MILP seed {args.seed}'}))
            os.replace(temp,dest)
        if i%25==0 or i==len(ids): print(f'Regional MILP: {i}/{len(ids)} puzzles, {saved} saved.',flush=True)
    print(dict(metrics),flush=True)

if __name__=='__main__': main()
