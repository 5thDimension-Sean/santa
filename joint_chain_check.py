"""Try a four-operation arithmetic chain after each six-output partial program."""
import json,subprocess,tempfile
from pathlib import Path
from joint_duration import banks,cut,OUT
from joint_near_repair import closure
from shared_device_bench import graph,visit,compile_graph,replay

def finish(p,keep,ports,seed):
    rules=keep|seed
    free=closure(p,rules);missing=ports-free
    if len(missing)!=1:raise ValueError('Expected one missing boundary output')
    t,=missing
    text=' '.join(map(str,[len(free),p.limit,t,*sorted(free)]))+'\n'
    exe=Path(tempfile.gettempdir())/'santa-joint-chain.exe'
    raw=subprocess.run([exe],input=text,text=True,capture_output=True,check=True)
    best=None;winner=None;n=0
    for line in raw.stdout.splitlines():
        x=line.split();keys=list(map(int,x[:4]))
        patch={k:(x[j],int(x[j+1]),int(x[j+2])) for k,j in zip(keys,range(4,16,3))}
        if any(v in rules for v in patch):continue
        n+=1
        merged=rules|patch;order=visit(p.targets,merged)
        if best is not None and len(order)>=len(best):continue
        c=compile_graph(p,merged,order)
        if c is not None:replay(p,c);best=c;winner=patch
    return {'target':t,'constructible_chains':n,'score':len(best) if best else None,'winning_chain':winner,'program':best}

def main():
    P,O,N=banks();p=P['scored_0379'];keep,ports,gone=cut(p,O[p.id],N[p.id])
    j=json.loads((OUT/'joint-two-sided-oracle.json').read_text())
    results=[]
    def record(label,seed):
        res=finish(p,keep,ports,{int(v):tuple(r) for v,r in seed.items()})
        res['source']=label;results.append(res)
    record('historical_six',j['arms'][0]['seed'])
    for i,x in enumerate(json.loads((OUT/'joint-six-states.json').read_text())):
        if len(x['missing'])==1:record('search_state_'+str(i),x['patch'])
    (OUT/'joint-chain-results.json').write_text(json.dumps(results))
    print([(r['source'],r['target'],r['constructible_chains'],r['score']) for r in results])
if __name__=='__main__':main()
