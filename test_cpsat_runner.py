"""Lightweight regression checks for the CP-SAT experiment runner."""
import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cpsat_experiments as E
import cpsat_lns as L
from elf import Puzzle


class RunnerTests(unittest.TestCase):
    def test_original_order_wins_tie(self):
        puzzle = Puzzle("toy", [3, 2, 4, 1], [5], 50, 4)
        prog = [["+", 0, 1, 0]]
        row = {}
        new = L.full_rebuild(puzzle, prog, {5: [("+", 1, 4)]}, None, None, 1, row)
        E.verify(puzzle, new)
        self.assertEqual(L.canonical(L.prog_graph(puzzle, new)[0][5]), ("+", 2, 3))

    def test_register_schedule_for_open_retries(self):
        import csv
        failed = {r["id"] for r in map(json.loads,(L.WORK/"exp0.jsonl").read_text().splitlines())
                  if r["base"] != r["new_len"]}
        puzzles = L.load_puzzles(str(L.ROOT/"puzzles.csv"))
        with L.BASE.open() as stream:
            for row in csv.DictReader(stream):
                if row["id"] not in failed:
                    continue
                program = json.loads(row["program"])
                p = puzzles[row["id"]]
                graph, order = L.prog_graph(p,program)
                rules = {v:L.canonical(r) for v,r in L.dce(p,graph).items()}
                scheduled = L.schedule(p,rules,{v:i for i,v in enumerate(order)},program)
                E.verify(p,scheduled)
                self.assertEqual(len(scheduled),len(program))

    def test_native_generators(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(L, "WORK", Path(temp)):
            helper = str(E.build_helper())
            for mode in ("mitm", "finisher"):
                with self.subTest(mode=mode):
                    c = L.Cands(100, {2, 7})
                    args = argparse.Namespace(native_source=mode, native_seconds=3,
                                              chain_cap=4, helper=helper)
                    L.native_chains(c,args,[2,7],[33])
                    p = Puzzle("toy",[2,7],[33],100,8)
                    rules,_,_ = L.solve_model(p,{2:None,7:None},c.rec,{33},{},1)
                    self.assertTrue(all(tag==mode for _,tag in rules.values()))
                    E.verify(p,L.schedule(p,{v:r for v,(r,_) in rules.items()},{}))

    def test_ratio_and_exact_division(self):
        args = E.parser().parse_args(["exp2"])
        args.ratio, args.residual, args.hubs, args.native_source = True, False, 0, ""
        puzzle = Puzzle("toy",[11,3,2,1],[11,17],100,8)
        c = L.generate(puzzle,dict.fromkeys(puzzle.starts),{17},{},{},args,random.Random(1))
        self.assertEqual(c.rec[17][("/",34,2)], "ratio")
        c.add(13,"/",27,2,"invalid")
        self.assertNotIn(("/",27,2),c.rec.get(13,{}))
        self.assertEqual(L.one_step_hits([10**17],[10**17],[1,10**17],10**18).count(
            (10**17,"/",10**17,1)),1)

    def test_historical_retry_and_gate(self):
        import csv
        with L.BASE.open() as stream:
            base = {r["id"]:json.loads(r["program"]) for r in csv.DictReader(stream)}
        with tempfile.TemporaryDirectory() as temp, patch.object(E,"DB",Path(temp)/"test.db"):
            with E.connect() as db:
                E.initialize(db)
                E.import_exp0(db,base,True)
                E.import_exp0(db,base,True)
                stats = E.metrics(db,0,"bank")
                self.assertEqual((stats["puzzles"],stats["matched"]),(1000,992))
                self.assertFalse(E.gate(0,{"bank":stats}))

    def test_checkpoint_resume_and_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (patch.object(E,"DB",root/"test.db"),patch.object(L,"ROOT",root),
                  patch.object(L,"EXP",root),patch.object(L,"OUT",root),patch.object(L,"BASE",root/"base.csv")):
                p = Puzzle("toy",[2,3],[5],100,4)
                prog = [["+",0,1,2]]
                args = E.parser().parse_args(["exp1"])
                with E.connect() as db:
                    E.initialize(db)
                    row = dict(stage=1,arm="control",id=p.id,iteration=0,base=2,
                               new_len=1,accepted=1,strategy="output",cpu_s=1)
                    E.record(db,row,prog)
                    E.run_puzzle((1,"control",p,prog,1,root,args))
                    self.assertEqual(E.metrics(db,1,"control")["saved"],1)
                    E.export_logs(db)
                    self.assertTrue((root/"cpsat-lns-improvements.csv").exists())
                    puzzles={str(i):Puzzle(str(i),p.starts,p.targets,100,4) for i in range(1000)}
                    base=dict.fromkeys(puzzles,prog)
                    L.BASE.with_suffix(".sources.csv").write_text("id,instructions,origin\n"+
                        "".join(f"{k},1,test\n" for k in puzzles))
                    self.assertEqual(E.publish(db,puzzles,base),1000)
                    before=(root/"submission-cpsat.csv").read_bytes()
                    with patch.object(E,"independent_check",return_value=(False,"fixture")):
                        with self.assertRaisesRegex(ValueError,"fixture"):
                            E.publish(db,puzzles,base)
                    self.assertEqual((root/"submission-cpsat.csv").read_bytes(),before)


if __name__=="__main__":
    unittest.main()
