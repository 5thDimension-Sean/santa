"""Independently repeat the notebook's two-step audit using register write IDs."""
from collections import Counter
from math import isqrt
from optimize import load_puzzles, read_bank

def reachable(snapshot,t,limit):
    live=sorted({x for x in snapshot if x is not None})
    if t in live: return 0
    one={1}
    for i,a in enumerate(live):
        for b in live[:i+1]:
            if a+b<=limit: one.add(a+b)
            if a*b<=limit: one.add(a*b)
            if a>b: one.add(a-b)
            if a%b==0: one.add(a//b)
    if t in one: return 1
    root=isqrt(t)
    if (t%2==0 and t//2 in one) or (root*root==t and root in one): return 2
    for x in live:
        if (t-x in one or t+x in one or x-t in one or
            (t%x==0 and t//x in one) or t*x in one or (x%t==0 and x//t in one)):
            return 2
    return 3

def audit(p,prog):
    regs=list(p.starts)+[None]*(p.registers-len(p.starts))
    writer=[-1]*p.registers
    snapshots=[]; parents=[]; consumers=Counter(); values=[]; roots=[]
    targets=set(p.targets); missing=targets-set(p.starts)
    for step,(op,i,j,d) in enumerate(prog):
        snapshots.append(tuple(regs))
        x,y=regs[i],regs[j]
        v=x+y if op=='+' else x-y if op=='-' else x*y if op=='*' else x//y
        deps={writer[i],writer[j]}-{-1}
        consumers.update(deps)
        parents.append(deps)
        regs[d],writer[d]=v,step
        values.append(v)
        if v in missing: missing.remove(v);roots.append(step)
    out=Counter()
    for root in roots:
        removed=Counter()
        stack=[root];size=0;start=root
        while stack:
            idx=stack.pop();size+=1;start=min(start,idx)
            for dep in parents[idx]:
                removed[dep]+=1
                if removed[dep]==consumers[dep] and values[dep] not in targets: stack.append(dep)
        if size>=3: out[size,reachable(snapshots[start],values[root],p.limit)]+=1
    return out

if __name__=='__main__':
    puzzles=load_puzzles('puzzles.csv'); programs=read_bank('submission.csv',puzzles)
    sample=Counter();all_rows=Counter()
    for i,k in enumerate(sorted(puzzles)):
        counts=audit(puzzles[k],programs[k]);all_rows.update(counts)
        if i%10==0: sample.update(counts)
    for label,counts in [('every tenth scored ID',sample),('all scored IDs',all_rows)]:
        print(label,'cones',sum(counts.values()),'within two',sum(n for (k,d),n in counts.items() if d<=2),
              'three-step cones with a three-step lower bound',counts[3,3],
              'longer cones',sum(n for (k,d),n in counts.items() if k>3))
