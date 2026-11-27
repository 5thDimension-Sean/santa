"""Bounded backward reductions ranked by the cost of their forward intersection."""
from math import gcd
import numpy as np


def canonical(r):
    op,a,b=r
    return (op,min(a,b),max(a,b)) if op in '+*' else r


def convergents(n,d,max_denominator):
    a,b,c,e=0,1,1,0
    while d:
        q,r=divmod(n,d)
        a,b,c,e=c,e,a+q*c,b+q*e
        if e>max_denominator: break
        yield c,e
        n,d=d,r


def bezout(a,b):
    x,y,u,v=1,0,0,1
    while b:
        q,r=divmod(a,b)
        a,b=b,r
        x,u=u,x-q*u
        y,v=v,y-q*v
    return a,x,y


def generate(t,live,vals,limit):
    free=set(map(int,live))
    def cheap(x):
        i=vals.searchsorted(x)
        return i<len(vals) and vals[i]==x
    offers={}
    def add(v,k,op,reverse,method):
        if not 0<v<=limit or v==t or k==t: return
        r=(op,k,v) if reverse else (op,v,k)
        key=canonical(r)
        # Keep the orientation with the cheapest outer operand.
        priority=(k not in free,k)
        if key not in offers or priority<offers[key][0]:
            offers[key]=(priority,(v,k,r,method))
    def offsets(k,method):
        add(t-k,k,'+',False,method);add(t+k,k,'-',False,method)
        add(k-t,k,'-',True,method)
    for k in sorted(free|{v for v in range(1,33) if cheap(v)}):
        offsets(k,'offset')
        if t%k==0: add(t//k,k,'*',False,'factor')
        if t<=limit//k: add(t*k,k,'/',False,'quotient')
        if k%t==0: add(k//t,k,'/',True,'quotient')
    # d*T = n*b +/- e. A non-unit denominator costs an exact division.
    # Only retain forms whose coefficients and correction are constructible.
    for b in sorted(free):
        nearest=(t+b//2)//b
        ratios=[(nearest,1),*convergents(t,b,12)]
        for n,d in ratios:
            if not n or n>limit//b or d>limit//t: continue
            e=abs(d*t-n*b)
            if e==0 or not cheap(n) or not cheap(e): continue
            if d==1: offsets(e,'nearest multiple' if n==nearest else 'convergent')
            elif cheap(d): add(d*t,d,'/',False,'convergent')
    # Bounded two-term integer solutions for the smaller available bases.
    small=sorted(free)[:10]
    for i,a in enumerate(small):
        for b in small[i+1:]:
            g=gcd(a,b)
            if t%g: continue
            _,x,y=bezout(a,b)
            x*=t//g;y*=t//g
            stride=b//g; shift=(-x)//stride
            for z in (shift,shift+1):
                u=x+z*stride;v=y-z*(a//g)
                if not (0<abs(u)<=32 and 0<abs(v)<=32): continue
                k=abs(u*a)
                if k<=limit and cheap(k): offsets(k,'two-term')
    return [item for _,item in offers.values()]


def rank(engine,t,live,vals,cost,helper_rows,limit,forbidden,oracle=()):
    proposed=generate(t,live,vals,limit)
    lookup={canonical(r) for _,_,r,_ in proposed}
    for v,k,r in oracle:
        if canonical(r) not in lookup: proposed.append((v,k,r,'injected'))
    proposed=[p for p in proposed if p[0] not in forbidden]
    if not proposed: return []
    residuals=np.array([x[0] for x in proposed],dtype=np.int64)
    m=len(proposed);helpers=vals[helper_rows]
    out=engine.run('frontier_links',len(helpers)*m,vals,cost,np.int32(len(vals)),
                   helpers,np.int32(len(helpers)),residuals,np.int32(m),np.int64(limit),
                   out=np.empty((len(helpers),m),dtype=np.int32))
    valid=out>=0
    comp=np.maximum(out,0)//6
    prices=1+cost[helper_rows,None]+np.where(comp==helper_rows[:,None],0,cost[comp])
    prices=np.where(valid,prices,127)
    best=prices.min(axis=0)
    indices=vals.searchsorted(residuals)
    exists=(indices<len(vals)) & (vals[np.minimum(indices,len(vals)-1)]==residuals)
    best=np.minimum(best,np.where(exists,cost[np.minimum(indices,len(vals)-1)],127))
    free=set(map(int,live))
    rows=[]
    for i,(v,k,r,method) in enumerate(proposed):
        # Count the final operation, the offset, and the estimated residual cost.
        # Unmatched residuals sort behind concrete joins.
        estimate=int(best[i])+1+(k not in free)
        rows.append(((estimate,abs(v-t).bit_length(),k,canonical(r)),v,k,r,method,int(best[i])))
    return sorted(rows)

def unfocused_rows(cost):
    return np.concatenate((np.flatnonzero(cost==0),np.flatnonzero(cost!=0)[:64]))
