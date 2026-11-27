"""Trace the rejection actually reached by each candidate in the historical repair path."""
from collections import Counter
import csv,inspect,random,sys
from joint_duration import banks,OUT
from near_rank_audit import available_table
from shared_device_bench import Engine,graph,repairs
def main():
    P,O,N=banks();eng=Engine(False)
    src,start=inspect.getsourcelines(repairs);line={s.strip():i+start for i,s in enumerate(src)}
    calls=list(csv.DictReader((OUT/'hub-oracle-leaf.csv').open()))
    failures=[r for r in calls if int(r['old_calls']) and not(int(r['old_replay']) or int(r['additional_replay']))]
    assert len(failures)==1018
    def profile(frame,event,arg):
        if frame.f_code is not repairs.__code__:return None
        if event!='line':return profile
        local=frame.f_locals;n=frame.f_lineno
        if n==line['if allowance<0 and collect_patch is None: return'] and local['allowance']<0:counts['budget']+=1
        if n==line['if not probes: return'] and not local['probes']:counts['budget']+=1
        if n==line['if len(patch)-len(roots)>allowance: continue'] and len(local['patch'])-len(local['roots'])>local['allowance']:counts['patch over allowance']+=1
        if n==line['if len(order)>len(prog): continue'] and len(local['order'])>len(local['prog']):counts['length']+=1
        if n==line['if len(order)==len(prog) and all(v in nodes for v in order): continue']:
            if len(local['order'])==len(local['prog']) and all(v in local['nodes'] for v in local['order']):counts['unchanged neutral']+=1
        if n==line['if candidate is None: continue'] and local['candidate'] is None:counts['register pressure']+=1
        if n==line['if not all(t in patch for t in roots): continue'] and not all(v in local['patch'] for v in local['roots']):counts['operand unavailable']+=1
        if n==line['if any(v in patch for v,_,_ in suffix): continue'] and any(v in local['patch'] for v,_,_ in local['suffix']):counts['circular recipe']+=1
        if n==line['if any(v in r[1:] for v,r in patch.items()): continue'] and any(v in r[1:] for v,r in local['patch'].items()):counts['circular recipe']+=1
        return profile
    output=[]
    for r in failures:
        k=r['id'];p=P[k];old=O[k];g=graph(p,old);t=int(r['target']);truth=graph(p,N[k])[t]
        _,free,ban,_,vs,_=available_table(eng,p,g,t);pool=set(map(int,vs))
        counts=Counter();attempts=0
        for v,x in ((truth[1],truth[2]),(truth[2],truth[1])):
            if x not in pool or v in ban:continue
            attempts+=1
            sys.settrace(profile)
            try:
                cand=next(repairs(eng,p,old,g,(t,),float('inf'),[0,0],random.Random(1),((t,v,truth),)),None)
            finally:sys.settrace(None)
            assert cand is None
        assert attempts==int(r['old_calls'])
        if int(r['fit_length'])>int(r['scheduled']):counts['register pressure']+=int(r['fit_length'])-int(r['scheduled'])
        if int(r['arithmetic_witnesses'])>int(r['fit_length']):counts['length']+=int(r['arithmetic_witnesses'])-int(r['fit_length'])
        # Attribute a case to the first failure on its furthest-reaching candidate.
        # A leaf depth exclusion alone does not invalidate a shallower generated witness.
        reason=next((key for key in ['register pressure','unchanged neutral','length','patch over allowance','circular recipe','operand unavailable','budget'] if counts[key]),'no residual witness in searched depth')
        output.append([k,t,reason,attempts,r['leaf_calls'],counts['length'],counts['register pressure'],counts['unchanged neutral'],counts['budget'],counts['patch over allowance'],counts['operand unavailable'],counts['circular recipe'],sum(x in free for x in truth[1:]),sum(x in pool for x in truth[1:])])
    with (OUT/'reroute-oracle-failures.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','target','case_reason','shallow_calls','leaf_calls','length_rejections','register_rejections','unchanged_neutral','budget_rejections','patch_over_allowance','operand_missing','circular_recipes','operands_in_snapshot','operands_within_one']);w.writerows(output)
    print(Counter(x[2] for x in output),flush=True)
if __name__=='__main__':main()
