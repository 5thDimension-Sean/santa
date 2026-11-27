import json,time
from joint_duration import banks,cut,search,OUT
from shared_device_bench import Engine,graph,visit,compile_graph,replay
from joint_near_repair import closure

def main():
    P,O,N=banks();p=P['scored_0379'];old=O[p.id];new=N[p.id]
    keep,ports,gone=cut(p,old,new);h=graph(p,new)
    engine=Engine(False)
    _,_,_=keep,ports,gone
    base=search(engine,p,old,keep,ports,gone,60,detail=True)
    six=[x for x in base['six_states'] if len(x['missing'])==1]
    if not six:raise RuntimeError('No six-output state found')
    # Prefer the first missing output observed, then the least expensive way to that partition.
    missing=six[0]['missing'][0]
    state=min((x for x in six if x['missing']==[missing]),key=lambda x:x['new_nodes'])
    (OUT/'joint-six-states.json').write_text(json.dumps(base['six_states']))
    actual=set(p.targets);results=[]
    def run(label,seed,required):
        free=closure(p,keep|seed)
        r=search(engine,p,old,keep,required,gone,60,seed=seed,detail=False)
        wins=[(round(t,5),replay(p,c)) for t,c in r['wins']]
        results.append({'arm':label,'seed':seed,'covered_ports_at_start':sorted(ports&free),
                        'required':sorted(required),'covered_required_at_start':len(required&free),
                        'max_required':max(n for _,n in r['progress']),
                        'elapsed':r['elapsed'],'exhausted':r['exhausted'],'wins':wins,
                        'over_old_cap':r['over_old_cap'],'pruned_by_cost':r['pruned_by_cost']})
        print(label,len(seed),len(ports&free),results[-1]['max_required'],len(required),wins,flush=True)
    # Intermediates that feed the seeded outputs are themselves paid instructions.
    for title,selected in [('historical_six',ports-{missing}),('historical_seventh',{missing})]:
        ancestors=visit(selected,h)
        seed={v:h[v] for v in ancestors if v not in keep}
        run(title+'_with_ancestors',seed,ports)
        seed={v:h[v] for v in selected}
        required=ports|({x for r in seed.values() for x in r[1:]}-set(keep)-set(seed)-set(p.starts))
        run(title+'_final_ops_only',seed,required)
    run('search_six_frozen',state['patch'],ports)
    truth=keep|{v:r for v,r in h.items() if v not in keep}
    full=compile_graph(p,truth,visit(p.targets,truth));assert replay(p,full)==108
    report={'missing':missing,'ports':sorted(ports),'deleted':len(gone),'six_states':len(six),
            'base_progress':base['progress'],'base_over_cap':base['over_old_cap'],
            'base_savings':len(old)-min([len(old)]+[len(c) for _,c in base['wins']]),
            'full_oracle_score':len(full),'arms':results}
    (OUT/'joint-two-sided-oracle.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
