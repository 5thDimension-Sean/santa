"""Local-only search. Run from the project directory. Uses no network services."""
import argparse
import csv
import hashlib
import json
import msvcrt
import os
import random
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from optimize import SOURCES, encode, load_puzzles, read_bank, replay, save

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'experiments'
ENGINES = ('joint', 'regional', 'frontier', 'products', 'seeded', 'factors', 'triwalk')

def pack(program):
    return json.dumps(program, separators=(',', ':'))

def block(program):
    return ['%s %d %d %d' % tuple(ins) for ins in program]

def decode(lines):
    result = []
    for line in lines:
        op,a,b,d = line.split()
        result.append([op,int(a),int(b),int(d)])
    return result

def search(engine,p,base,alternates,seed,epoch):
    rng = random.Random(seed)
    alt = rng.choice(alternates) if alternates else base
    initial = alt if alternates and rng.random() < 0.45 else base
    seconds = min(14, 1.5 + epoch*0.75)
    if engine in ('regional','products','factors'):
        from region_milp import proposals
        for candidate in proposals(p,base,alternates,rng,backbone=engine=='regional',
                                   products=engine=='products',factors=engine=='factors'):
            return candidate
        return base
    elif engine == 'seeded':
        programs = [base] + rng.sample(alternates,min(5,len(alternates)))
        rounds = min(2000,120+epoch*75)
        rows = ['1',f'{p.limit} {p.registers} {rounds} 0 {seed}',
                ' '.join(map(str,[len(p.starts),*p.starts])),
                ' '.join(map(str,[len(p.targets),*p.targets])),str(len(programs))]
        for prog in programs:
            rows.extend([str(len(prog)),*block(prog)])
        args = [WORK/'seeded.exe']
    elif engine == 'triwalk':
        args = [ROOT/'triwalk.exe','--time',str(seconds),'--seed',str(seed),
                '--cap',str(rng.choice([1,2,3])),'--T','0.2','--rb','0.75','--deep','0.2']
        rows = [encode(p,initial)]
    else:
        rows = [f'{p.id} {p.limit} {len(p.starts)} {len(p.targets)} {len(initial)}',
                ' '.join(map(str,p.starts)),' '.join(map(str,p.targets)),*block(initial)]
        if engine=='joint':
            args=[WORK/'joint.exe','12',str(rng.choice([8,16,32])),str(rng.choice([1,2])),str(seed),str(int(1000*seconds))]
        else:
            args=[WORK/'frontier.exe','10','5600','10','11',str(rng.choice([9,11,13])),str(seed),str(int(1000*seconds))]
    run=subprocess.run(args,input='\n'.join(rows)+'\n',text=True,capture_output=True,
                       timeout=max(60,seconds*10),check=True)
    lines=run.stdout.splitlines()
    if engine=='triwalk':
        if not lines:
            if f'{p.id} {len(initial)} ->' not in run.stderr: raise ValueError(run.stderr[-150:])
            return initial
        status,key,old,count=lines[0].split()
        if status not in {'R','V'} or key!=p.id or int(old)!=len(initial): raise ValueError(lines[0])
        candidate=decode(lines[1:])
    elif engine=='seeded':
        tokens=run.stdout.split()
        count=tokens[0]
        candidate=decode([' '.join(tokens[i:i+4]) for i in range(1,len(tokens),4)])
    else:
        key,count,*_=lines[0].split()
        if key!=p.id: raise ValueError(lines[0])
        candidate=decode(lines[1:])
    if int(count)!=len(candidate): raise ValueError('Engine returned incomplete program')
    replay(p,candidate)
    return candidate

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target',type=int,default=124177)
    ap.add_argument('--jobs',type=int,default=6)
    args=ap.parse_args()
    os.chdir(ROOT)
    lock=(WORK/'local-search.lock').open('a+b')
    if not lock.tell(): lock.write(b'0');lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    puzzles=load_puzzles('puzzles.csv')
    best=read_bank('submission.csv',puzzles,strict=True)
    with open('submission.sources.csv',newline='') as stream:
        origins={r['id']:r['origin'] for r in csv.DictReader(stream)}
    db=sqlite3.connect(WORK/'local-search.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS wins (id TEXT PRIMARY KEY, program TEXT, origin TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS batches (batch INTEGER PRIMARY KEY, engine TEXT, tried INTEGER, errors INTEGER, gain INTEGER, score INTEGER, elapsed REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS variants (id TEXT, digest TEXT, program TEXT, PRIMARY KEY(id,digest))')
    db.commit()
    for k,s,origin in db.execute('SELECT id,program,origin FROM wins'):
        prog=json.loads(s)
        if replay(puzzles[k],prog)<len(best[k]): best[k],origins[k]=prog,origin
    score=save('submission.csv',puzzles,best,origins)
    active=[k for k,p in puzzles.items() if p.limit>16 and len(best[k])>len(set(p.targets)-set(p.starts))]
    banks=[read_bank(s,puzzles) for s in SOURCES]
    batch=db.execute('SELECT COALESCE(MAX(batch),-1)+1 FROM batches').fetchone()[0]
    batch_size=32
    slices=(len(active)+batch_size-1)//batch_size
    failures=0
    imported={}
    print(f'Local search running. PID {os.getpid()}, score {score}, target {args.target}, workers {args.jobs}.',flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        while score>args.target:
            changed=False
            for path in WORK.glob('region-*.json'):
                stamp=path.stat().st_mtime_ns
                if imported.get(path)==stamp: continue
                record=json.loads(path.read_text())
                k,prog,origin=record['id'],record['program'],record['origin']
                if replay(puzzles[k],prog)<len(best[k]):
                    with db: db.execute('INSERT OR REPLACE INTO wins VALUES (?,?,?)',(k,pack(prog),origin))
                    score-=len(best[k])-len(prog)
                    best[k],origins[k]=prog,origin
                    changed=True
                imported[path]=stamp
            if changed:
                if save('submission.csv',puzzles,best,origins)!=score: raise RuntimeError('Import score mismatch')
                if score<=args.target: break
            epoch,which=divmod(batch,len(ENGINES)*slices)
            section,method=divmod(which,len(ENGINES))
            engine=ENGINES[method]
            keys=active.copy()
            random.Random(5821+epoch*17+method).shuffle(keys)
            keys=keys[section*batch_size:(section+1)*batch_size]
            before=score
            started=time.monotonic()
            jobs=[]
            for k in keys:
                base=best[k]
                slack=28 if engine in ('regional','products','factors') else 4
                alts=[b[k] for b in banks if len(b[k])<=len(base)+slack]
                alts.extend(json.loads(s) for s, in db.execute('SELECT program FROM variants WHERE id=?',(k,)))
                unique={pack(a):a for a in alts if len(a)<=len(base)+slack}
                unique.pop(pack(base),None)
                seed=(batch*1009+int(k[-4:])*2347+1559)&0x7fffffff
                jobs.append((k,seed,pool.submit(search,engine,puzzles[k],base,list(unique.values()),seed,epoch)))
            errors=0
            for k,seed,future in jobs:
                try: prog=future.result()
                except (ValueError,IndexError,subprocess.SubprocessError,OSError) as exc:
                    errors+=1
                    if errors==1: print(f'{engine} {k}: {exc}',flush=True)
                    continue
                content=pack(prog)
                if len(prog)<len(best[k]):
                    origin=f'local {engine} seed {seed}, starting bank and variants in {WORK.name}'
                    with db: db.execute('INSERT OR REPLACE INTO wins VALUES (?,?,?)',(k,content,origin))
                    score-=len(best[k])-len(prog)
                    best[k],origins[k]=prog,origin
                elif prog!=best[k] and len(prog)<=len(best[k])+1:
                    digest=hashlib.sha256(content.encode()).hexdigest()
                    with db:
                        db.execute('INSERT OR IGNORE INTO variants VALUES (?,?,?)',(k,digest,content))
                        db.execute('DELETE FROM variants WHERE id=? AND rowid NOT IN (SELECT rowid FROM variants WHERE id=? ORDER BY length(program),rowid DESC LIMIT 12)',(k,k))
            if score<before:
                checked=save('submission.csv',puzzles,best,origins)
                if checked!=score: raise RuntimeError('Checkpoint score mismatch')
            with db:
                db.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)',(batch,engine,len(keys),errors,before-score,score,time.monotonic()-started))
            print(f'Batch {batch} {engine}: {len(keys)} puzzles, {errors} errors, saved {before-score}, best {score}.',flush=True)
            failures=failures+1 if errors==len(keys) else 0
            if failures>=len(ENGINES): raise RuntimeError('All search methods failed')
            batch+=1
    result=subprocess.run(['node','benchmark.cjs','--verify','puzzles.csv','submission.csv'],check=True,capture_output=True,text=True)
    audit=json.loads(result.stdout)
    if audit['verified']!=1000 or audit['instructions']!=score: raise RuntimeError('Independent checkpoint mismatch')
    print(result.stdout.strip(),flush=True)

if __name__=='__main__': main()
