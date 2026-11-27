import ctypes
import os
from pathlib import Path
import tempfile
from collections import Counter
import numpy as np
from joint_duration import banks,cut,OUT
from joint_near_repair import closure

def forward(free,L):
    out={};vs=sorted(free)
    for i,a in enumerate(vs):
        for b in vs[:i+1]:
            for v,r in ((a+b,('+',a,b)),(a-b,('-',a,b)),(a*b,('*',a,b)),(a//b if a%b==0 else 0,('/',a,b))):
                if 0<v<=L and v not in free:out.setdefault(v,r)
    return out

def score(free,candidates,targets,limit):
    global dll
    if 'dll' not in globals():dll=ctypes.CDLL(str(Path(tempfile.gettempdir())/'santa-hub-vote.dll'))
    arrays=[np.array(sorted(group),dtype=np.int64) for group in (free,candidates,targets)]
    a,b,t=arrays
    costs=np.empty((len(b),len(t)),dtype=np.uint8);res=np.empty(costs.shape,dtype=np.int64)
    ptr=lambda v:ctypes.c_void_p(v.ctypes.data)
    dll.vote(ptr(a),len(a),ptr(b),len(b),ptr(t),len(t),ctypes.c_int64(limit),ptr(costs),ptr(res))
    votes=np.where(costs<=4,1/(1+costs.astype(float)),0).sum(axis=1)
    return [(int(b[i]),float(votes[i]),dict(zip(t.tolist(),costs[i].tolist())),dict(zip(t.tolist(),res[i].tolist()))) for i in np.lexsort((b,-votes))]

if __name__=='__main__':
    import json
    P,O,N=banks();p=P['scored_0379'];keep,ports,_=cut(p,O[p.id],N[p.id]);f=closure(p,keep)
    pool=forward(f,p.limit);v=score(f,pool,ports,p.limit)
    print('free',len(f),'candidates',len(pool),'rank',next((i for i,r in enumerate(v,1) if r[0]==787994),None))
    (OUT/'hub-start-votes.json').write_text(json.dumps([(b,s,c) for b,s,c,_ in v]))

def exact_two(free,targets,limit):
    global distances
    if 'distances' not in globals():distances=ctypes.CDLL(str(Path(tempfile.gettempdir())/'santa-hub-distance.dll'))
    v=np.array(sorted(free),dtype=np.int64);t=np.array(sorted(targets),dtype=np.int64)
    result=np.empty(len(t),dtype=np.uint8)
    ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
    distances.distance2(ptr(v),len(v),ptr(t),len(t),ctypes.c_int64(limit),ptr(result))
    return dict(zip(t.tolist(),result.tolist()))
