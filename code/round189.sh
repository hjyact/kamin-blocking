#!/bin/bash
# baselines at step 800 for the unblocking test (round188): blocked (no shift) and control without pull, same seed; DLA via strength.py
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
until grep -q DONE /tmp/claude-1000/-home-anon/88c8e374-4331-462b-8701-47d9a36fe4a0/tasks/bbc03s8bz.output 2>/dev/null; do sleep 30; done
B="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 300,800,2400 --nosave"
$P train.py --name K2CPREb $B --nudge 0.05 --nudge_spec "3,2,0,300" > logs/K2CPREb_s0.log 2>&1
$P train.py --name K2CTLb $B > logs/K2CTLb_s0.log 2>&1
for n in K2CPREb K2CTLb K2UP3 K2DN3 K2RT3 K2CTLUP3; do
  echo $n >> logs/round189.txt
  WATCH="3,2;6,1" $P winner.py runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep top3 >> logs/round189.txt
  HEADS="3,2;6,1" $P strength.py runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep DLA >> logs/round189.txt
done
for n in KAUP3 KADN3; do
  echo $n >> logs/round189.txt; HEADS="4,4;5,6" $P strength.py runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep DLA >> logs/round189.txt
done
echo DONE
