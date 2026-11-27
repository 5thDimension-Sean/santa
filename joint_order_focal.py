"""Focal ordering experiment. Enumerate cheap corrections of a multiple using four linear terms."""
import json
from joint_duration import banks,cut,search,OUT
from joint_near_repair import paths
from shared_device_bench import Engine,replay
from collections import defaultdict

def expanded_paths(engine,p,free,t):
    # Retain the old generator. Add (signed pair +/- signed pair)*b +/- c.
    yield from paths(engine,p,free,t)
    vals=sorted(free);pairs=defaultdict(list)
    for i,a in enumerate(vals):
        for b in vals[:i+1]:
            for z,r in ((a-b,('-',a,b)),(a+b,('+',a,b))):
                if 0<z<=p.limit and z!=t and len(pairs[z])<2:pairs[z].append(r)
    for c in vals:
        for product,outer in [(t+c,('-',t+c,c)),(t-c,('+',t-c,c)),(c-t,('-',c,c-t))]:
            if not 0<product<=p.limit or product in free:continue
            for b in vals:
                if b<2 or product%b:continue
                v=product//b
                if v in free or v in (t,product):continue
                for a,rs in pairs.items():
                    for z,op,rev in [(v-a,'+',False),(v+a,'-',True),(a-v,'-',False)]:
                        if z not in pairs or {z,a}&{t,product,v}:continue
                        x,y=(z,a) if rev else (a,z)
                        for ra in rs:
                            for rz in pairs[z]:
                                patch={a:ra,z:rz}
                                patch={u:r for u,r in patch.items() if u not in free}
                                patch.update({v:(op,x,y),product:('*',b,v),t:outer})
                                yield patch

def main():
    P,O,N=banks();p=P['scored_0379'];old=O[p.id];keep,ports,gone=cut(p,old,N[p.id])
    e=Engine(False);results=[]
    for name,func in [('output-order diversity',paths),('output-order diversity plus linear chains',expanded_paths)]:
        r=search(e,p,old,keep,ports,gone,60,detail=True,diverse=True,pathgen=func)
        wins=[(t,replay(p,c)) for t,c in r['wins']]
        if wins:(OUT/'joint-order-focal-program.json').write_text(json.dumps(r['wins'][-1][1]))
        results.append({'arm':name,'elapsed':r['elapsed'],'ports':r['progress'],'wins':wins,'exhausted':r['exhausted']})
        print(results[-1],flush=True)
    (OUT/'joint-order-focal.json').write_text(json.dumps(results))
if __name__=='__main__':main()
