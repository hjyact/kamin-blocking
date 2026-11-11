#!/bin/bash
# depth theory of the race: circuit = product of p factors -> dS/dt ~ S^(2-2/p) e; p>=3 winner-take-all, p=2 ratio-preserving (graded).
# keyshift heads (H>=4) do induction in one layer (no previous-token head): one composition shallower.
# predict: in keyshift pair (L3H6, L6H5) blocking is graded with head start (D falls smoothly with dose), not the sharp switch of pair 2 (D .246 at 50 -> .023 at 100).
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
until grep -q DONE /tmp/claude-1000/-home-anon/88c8e374-4331-462b-8701-47d9a36fe4a0/tasks/bnhruzpe3.output 2>/dev/null; do sleep 30; done
B="--seed 0 --tokens 100e6 --keyshift --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,6,300,800;6,5,300,800 --save_at 300,800,2400 --nosave"
$P train.py --name KSCTL $B > logs/KSCTL_s0.log 2>&1
for k in 50 100 300; do $P train.py --name KS$k $B --nudge 0.05 --nudge_spec "3,6,0,$k" > logs/KS${k}_s0.log 2>&1; done
for n in KSCTL KS50 KS100 KS300; do
  echo $n >> logs/round190.txt
  WATCH="3,6;6,5" LEDGER_ARGS="--keyshift" $P winner.py runs/${n}_s0_step300.pt runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep top3 >> logs/round190.txt
  HEADS="3,6;6,5" LEDGER_ARGS="--keyshift" $P strength.py runs/${n}_s0_step300.pt runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep DLA >> logs/round190.txt
  grep -h FINAL logs/${n}_s0.log >> logs/round190.txt
done
echo DONE
