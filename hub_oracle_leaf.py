"""Extend the prior forced-final-operation test with the four-instruction leaf."""
import csv,json,subprocess,tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from joint_duration import banks,OUT
from joint_near_repair import closure
from shared_device_bench import visit,graph,compile_graph,replay
from hub_vote import forward

def chain(free,limit,t):
    exe=Path(tempfile.gettempdir())/'santa-joint-chain.exe'
    inp=' '.join(map(str,[len(free),limit,t,*sorted(free)]))+'\n'
    run=subprocess.run([exe],input=inp,text=True,capture_output=True,check=True)
    for ln in run.stdout.splitlines():
        items=ln.split()
        yield {int(k):(items[j],int(items[j+1]),int(items[j+2])) for k,j in zip(items[:4],range(4,16,3))}

def evaluate(args):
    p,old,targets=args;g=graph(p,old);out=[]
    for t,truth,accepted,calls in targets:
        row=[p.id,t,calls,accepted,0,0,0,0,0,0]
        if accepted or not calls:out.append(row);continue
        keep=visit(p.targets,g,(t,));base={v:g[v] for v in keep}
        free=closure(p,base);one=forward(free,p.limit);banned=(set(base)-free)|{t}
        for residual,outer in ((truth[1],truth[2]),(truth[2],truth[1])):
            if outer not in free and outer not in one:continue
            if residual in banned or outer in banned:continue
            seed={} if outer in free else {outer:one[outer]}
            # A four-new-value leaf and the final operation require at least five steps.
            if len(keep)+5>len(old):continue
            row[4]+=1
            for leaf in chain(free|set(seed),p.limit,residual):
                if set(leaf)&(banned|set(seed)):continue
                row[5]+=1
                rules=base|seed|leaf|{t:truth}
                order=visit(p.targets,rules)
                if len(order)>len(old):continue
                row[6]+=1
                c=compile_graph(p,rules,order)
                if c is None:continue
                row[7]+=1;length=replay(p,c)
                if length==len(old) and not(set(order)-g.keys()):continue
                row[8]=1;row[9]=len(old)-length
                break
            if row[8]:break
        out.append(row)
    return out

def main():
    P,O,N=banks();labels={}
    truth={(r['id'],r['target']):(r['true_op'],int(r['a']),int(r['b'])) for r in csv.DictReader((OUT/'near-miss-truth-ranks.csv').open())}
    for r in csv.DictReader((OUT/'oracle-injection-calls.csv').open()):
        k=r['id'];t=r['target'];labels.setdefault(k,[]).append((int(t),truth[k,t],int(r['accepted']),int(r['calls'])))
    with (OUT/'hub-oracle-leaf.csv').open('w',newline='') as f,ProcessPoolExecutor(max_workers=4) as pool:
        w=csv.writer(f);w.writerow(['id','target','old_calls','old_replay','leaf_calls','arithmetic_witnesses','fit_length','scheduled','additional_replay','saved'])
        for rows in pool.map(evaluate,[(P[k],O[k],rs) for k,rs in labels.items()]):w.writerows(rows);f.flush()
if __name__=='__main__':main()
