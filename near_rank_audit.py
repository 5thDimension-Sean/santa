"""Checkpoint evaluation. Historical recipes enter only the labelled oracle path."""
import csv
import random
from statistics import median
from pathlib import Path
import numpy as np
from near_reductions import canonical, rank, unfocused_rows
from shared_device_bench import Engine, graph, load_puzzles, read_bank, repairs, replay, sample_suffixes, visit, TICK
from optimize import SOURCES

OUT=Path('experiments')

def available_table(engine,p,g,t):
    keep=visit(p.targets,g,(t,))
    free=set(p.starts)
    for v in keep:
        if all(x in free for x in g[v][1:]): free.add(v)
    forbidden=(set(keep)-free)|{t}
    live=np.array(sorted(free),dtype=np.int64); n=len(live)
    raw=engine.run('make_values',n*n,live,np.int32(n),np.int64(p.limit),out=np.empty(4*n*n,dtype=np.int64))
    vals=np.unique(np.concatenate((live,raw)))
    vals=vals[(vals>0)&~np.isin(vals,list(forbidden))]
    cost=(~np.isin(vals,live)).astype(np.int32)
    return keep,free,forbidden,live,vals,cost


def main():
    P=load_puzzles('puzzles.csv')
    old=read_bank('public/ishankarnick/submission.csv',P)
    current=read_bank('submission.csv',P,strict=True)
    merged={};source={}
    for path in SOURCES:
        bank=read_bank(path,P)
        for k,c in bank.items():
            if k not in merged or len(c)<len(merged[k]):
                merged[k]=c; source[k]=path
    assert all(len(current[k])==len(merged[k]) for k in P)
    licenses={'frontier-nov27':'Apache-2.0; CC0 upstream retained','ishankarnick':'Apache-2.0; CC0 upstream retained','b05204047elias':'Apache-2.0; CC0 upstream retained','viktorbarbarich':'CC0-1.0 (value-graph LNS dataset)','asterios':'Apache-2.0 (programs and experiments dataset)','alexchilton':'CC0-1.0','goodpjw2008':'Apache-2.0 (DAG-LNS dataset)','farshid':'CC0-1.0 (santa-2026-share; SHA-256 matched the published CSV)'}
    with (OUT/'near-miss-license-check.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','first_best_rows','license_evidence'])
        for path in SOURCES:w.writerow([path,list(source.values()).count(path),licenses[path.split('/')[1]]])
    print('Merged/replayed',sum(replay(P[k],c) for k,c in merged.items()),flush=True)
    engine=Engine(False); rows=[];puzzles=[]
    for k,p in P.items():
        delta=len(old[k])-len(current[k])
        if delta<=0:continue
        g=graph(p,old[k]);new=graph(p,current[k]); local=[]
        for t in p.targets:
            if t not in new or t not in g or canonical(g[t])==canonical(new[t]):continue
            keep,free,forbidden,live,vals,cost=available_table(engine,p,g,t)
            truth=canonical(new[t]);rng=random.Random(781+int(k[-4:])+p.targets.index(t))
            legacy=[]
            for suffix in sample_suffixes(t,free,p.limit,rng):
                if not suffix or len(suffix)>len(old[k])-len(keep)-1: continue
                if suffix[-1][1] in forbidden:continue
                r=canonical(suffix[0][2])
                if r not in legacy:legacy.append(r)
            ranked=rank(engine,t,live,vals,cost,unfocused_rows(cost),p.limit,forbidden)
            def exact(rows):
                helpers=vals[rows]
                result=engine.run('frontier_links',len(rows),vals,cost,np.int32(len(vals)),
                                  helpers,np.int32(len(helpers)),np.array([t],dtype=np.int64),
                                  np.int32(1),np.int64(p.limit),out=np.empty(len(rows),dtype=np.int32))
                unique={}
                for j in np.flatnonzero(result>=0):
                    qi,op=divmod(int(result[j]),6)
                    h,x=int(helpers[j]),int(vals[qi])
                    a,b=(x,h) if op in (2,5) else (h,x)
                    unique[canonical(('+--*//'[op],a,b))]=None
                return list(unique)
            old_exact=exact(np.arange(len(vals)))
            new_exact=exact(unfocused_rows(cost))
            def position(candidates):
                keys=list(dict.fromkeys(candidates))
                return keys.index(truth)+1 if truth in keys else None
            old_any=position(old_exact+legacy)
            new_any=position(new_exact+[canonical(x[3]) for x in ranked])
            newrank=next((i for i,x in enumerate(ranked,1) if canonical(x[3])==truth),None)
            oldrank=legacy.index(truth)+1 if truth in legacy else None
            # Force both orientations wherever the outer operand can be paid for.
            # Replay counts include neutral programs only if they use fresh values.
            live_one=set(map(int,vals));oracle='no eligible orientation in surviving pool'
            for v,x in ((new[t][1],new[t][2]),(new[t][2],new[t][1])):
                if x not in live_one or v in forbidden:continue
                oracle='injected; no accepted repair'
                cand=next(repairs(engine,p,old[k],g,(t,),float('inf'),[0,0],rng,((t,v,new[t]),)),None)
                if cand is not None:
                    replay(p,cand)
                    oracle='replayed';break
            passed=bool(newrank and newrank<=64 and ranked[newrank-1][-1]<127 and len(old[k])-len(keep)-1>=1)
            local.append((k,delta,t,*new[t],oldrank,newrank,passed,oracle,old_any,new_any,truth in old_exact,truth in new_exact))
        rows.extend(local);puzzles.append((k,delta,len(local)))
    with (OUT/'near-miss-truth-ranks.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','net_steps','target','true_op','a','b','legacy_backward_rank','new_backward_rank','passed_backward_beam_screen','oracle_result','old_exact_then_backward_rank','new_exact_then_backward_rank','legacy_direct','new_direct']);w.writerows(rows)
    grouped={}
    for row in rows: grouped.setdefault(row[0],[]).append(row)
    summaries=[]
    for k,delta,count in puzzles:
        group=grouped.get(k,[])
        ranks=[median([r[col] or float('inf') for r in group]) if group else '' for col in (10,11)]
        summaries.append((k,delta,count,*ranks,sum(bool(r[11]) and r[11]<=10 for r in group)))
    with (OUT/'near-miss-puzzle-coverage.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','net_steps','changed_target_recipes','old_median_exact_then_backward','new_median_exact_then_backward','new_top10']);w.writerows(summaries)
    for col in [6,7,10,11]:
        finite=[r[col] for r in rows if r[col]]
        print(col,'rows',len(rows),'present',len(finite),'finite median',median(finite),'all median',median([r[col] or float('inf') for r in rows]),'top 10',sum(x<=10 for x in finite),flush=True)
    print('in beam',sum(r[8] for r in rows),'oracle replayed',sum(r[9]=='replayed' for r in rows),'puzzles',len(puzzles),'net',sum(r[1] for r in puzzles),'no changed target',sum(r[2]==0 for r in puzzles))
if __name__=='__main__':main()
