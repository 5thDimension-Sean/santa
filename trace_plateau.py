"""Replay a historically known path through the candidate generator (diagnostic only)."""
from shared_device_bench import Engine, TICK, graph, load_puzzles, operands, read_bank, repairs, replay
import random

def main():
    puzzles=load_puzzles('puzzles.csv')
    k='scored_0379'
    p=puzzles[k]
    old=read_bank('public/ishankarnick/submission.csv',puzzles)[k]
    newer=read_bank('submission.csv',puzzles)[k]
    desired=graph(p,newer)
    for gpu in (False,True):
        engine=Engine(gpu)
        prog=old
        for root, outer, required in [
            (51307,[],[11827]),
            (92818019,[4692],[93610705,787994,92822711]),
            (747018334,[22],[948,747018312]),
            (168556460436,[2940,98448],[1707449,1712137,168556463376])
        ]:
            suffix=[]
            v=root
            for x in outer:
                r=desired[v]
                w=next(y for y in r[1:] if y!=x)
                assert any(w==z and (rr==r or (rr[0]==r[0] and r[0] in '+*' and set(rr[1:])==set(r[1:]))) for z,rr in operands(v,x,p.limit))
                suffix.append((v,w,r))
                v=w
            before=len(prog)
            for cand in repairs(engine,p,prog,graph(p,prog),(root,),TICK()+20,[0,0],random.Random(1),tuple(suffix)):
                if set(required)<=graph(p,cand).keys():
                    prog=cand
                    break
            else:
                raise ValueError(f'No witness at {root}')
            print(('CPU','GPU')[gpu],root,before,'->',replay(p,prog),flush=True)
        assert len(prog)==108

if __name__=='__main__': main()
