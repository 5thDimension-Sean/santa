"""Exact meet in the middle when the destination is the sole scratch register."""
from math import isqrt

def one_scratch(regs,d,target,limit,blocked,max_depth=5):
    op0,a0,b0=blocked
    constants={v:i for i,v in enumerate(regs) if v and i!=d}
    def legal(op,a,b):
        return not (op==op0 and ((a,b)==(a0,b0) or op in '+*' and (a,b)==(b0,a0)))
    def forward(a,i,b,j):
        for op,v,x,y in [('+',a+b,i,j),('*',a*b,i,j),('-',a-b,i,j),('-',b-a,j,i),
                          ('/',a//b if a%b==0 else 0,i,j),('/',b//a if b%a==0 else 0,j,i)]:
            l,r=(a,b) if (x,y)==(i,j) else (b,a)
            if 0<v<=limit and legal(op,l,r):yield v,(op,x,y,d)
    pool={**constants}
    if regs[d]:pool[regs[d]]=d
    first={}
    vals=list(pool.items())
    for pos,(a,i) in enumerate(vals):
        for b,j in vals[:pos+1]:
            for v,ins in forward(a,i,b,j):first.setdefault(v,ins)
    if target in first:return [first[target]],1
    def reverse(t):
        for x,j in constants.items():
            for v,op,rev in [(t-x,'+',False),(t+x,'-',False),(x-t,'-',True),
                             (t//x if t%x==0 else 0,'*',False),(t*x,'/',False),(x//t if x%t==0 else 0,'/',True)]:
                a,b=(x,v) if rev else (v,x)
                if 0<v<=limit and legal(op,a,b):yield v,(op,j if rev else d,d if rev else j,d)
        root=isqrt(t)
        for v,op,yes in [(t//2,'+',t%2==0),(root,'*',root*root==t)]:
            if yes and legal(op,v,v):yield v,(op,d,d,d)
    # The multiplication ablations reaching this fallback always have a
    # permanently readable register: self-division to 1 is a first-step reset.
    assert op0=='*' and constants
    back={target:()}
    for depth in (2,3):
        following={}
        for t,tail in back.items():
            for v,ins in reverse(t):
                if v in first:return [first[v],ins,*tail],depth
                following.setdefault(v,(ins,*tail))
        back=following
    # Match two forward instructions against the two-instruction reverse index.
    two={}
    for u,head in first.items():
        for x,j in [*constants.items(),(u,d)]:
            for v,ins in forward(u,d,x,j):
                if v in back:return [head,ins,*back[v]],4
                if max_depth>=5:two.setdefault(v,(u,ins))
    if max_depth>=5:
        for t,tail in back.items():
            for v,ins in reverse(t):
                if v in two:
                    u,second=two[v];return [first[u],second,ins,*tail],5
    return None,max_depth
