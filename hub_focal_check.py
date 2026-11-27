"""Only the post-search diagnostics know the focal puzzle's historical hub."""
import json
from hub_beam import build_library,run
from hub_vote import forward,score
from joint_duration import banks,cut,OUT
from joint_near_repair import closure
from joint_chain_check import finish
from shared_device_bench import replay

def main():
    P,O,N=banks();key='scored_0379';p=P[key];keep,ports,_=cut(p,O[key],N[key])
    free=closure(p,keep);candidates=forward(free,p.limit)
    ranks=score(free,candidates,ports,p.limit);watch=787994
    bank,conditional=build_library(p,keep,ports)
    gen,sur,states,best,wins,leaves=run(p,O[key],keep,ports,bank,conditional,'new',monitor=watch)
    # Reconstruct the partial program independently from serialized-friendly recipes.
    witness=None
    for state in states:
        if not state['observed_hub'] or state['cost']>14:continue
        patch=state['patch'];actual=closure(p,keep|patch)
        assert watch in actual and len(ports-actual)==1
        checked=finish(p,keep,ports,patch)
        if checked['score']==108:
            witness=state|{'last_chain':checked['winning_chain'],'score':replay(p,checked['program'])}
            break
    report={'initial_pool':len(free),'candidate_count':len(candidates),'initial_rank':next((i for i,r in enumerate(ranks,1) if r[0]==watch),None),
            'conditional_rank':next((i for i,r in enumerate(conditional,1) if r[2]==watch),None),
            'conditional_candidates':len(conditional),'six':states,'witness':witness,'leaf_checks':leaves,
            'wins':wins,'old_score':len(O[key]),'public_score':len(N[key]),'score':replay(p,best),'program':best}
    (OUT/'hub-focal-verified.json').write_text(json.dumps(report,indent=2))
    print({k:report[k] for k in ('initial_pool','candidate_count','initial_rank','conditional_rank','conditional_candidates','wins','score')},'verified',bool(witness),flush=True)
if __name__=='__main__':main()
