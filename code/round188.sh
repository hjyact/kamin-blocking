#!/bin/bash
# unblocking by outcome upshift vs downshift between heads (Holland 1984 test: RW vs Pearce-Hall).
# oread_T scales the shared readout's temperature at a fixed rate (grad = p_T - y): T=3 upshift, T=0.33 downshift; oread_rate 3 = rate-only control.
# RW predicts: upshift unblocks D (L6H1 at 800 > 0.10, blocked is 0.051), downshift keeps it blocked (< 0.05), rate-only does not unblock (< 0.08).
# Pearce-Hall predicts both upshift and downshift unblock.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
until grep -q DONE /tmp/claude-1000/-home-anon/88c8e374-4331-462b-8701-47d9a36fe4a0/tasks/bq5c7737p.output 2>/dev/null; do sleep 30; done
B="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 300,800,2400 --nosave"
$P train.py --name K2UP3 $B --nudge 0.05 --nudge_spec "3,2,0,300" --oread_T 3 > logs/K2UP3_s0.log 2>&1
$P train.py --name K2DN3 $B --nudge 0.05 --nudge_spec "3,2,0,300" --oread_T 0.333 > logs/K2DN3_s0.log 2>&1
$P train.py --name K2RT3 $B --nudge 0.05 --nudge_spec "3,2,0,300" --oread_rate 3 > logs/K2RT3_s0.log 2>&1
$P train.py --name K2CTLUP3 $B --oread_T 3 > logs/K2CTLUP3_s0.log 2>&1
A="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 4,4,300,800;5,6,300,800 --save_at 300,800,2400 --nosave --nudge 0.05 --nudge_spec 4,4,0,300"
$P train.py --name KAUP3 $A --oread_T 3 > logs/KAUP3_s0.log 2>&1
$P train.py --name KADN3 $A --oread_T 0.333 > logs/KADN3_s0.log 2>&1
for n in K2UP3 K2DN3 K2RT3 K2CTLUP3; do
  WATCH="3,2;6,1" $P winner.py runs/${n}_s0_step300.pt runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep -v Warn > logs/${n}_winner.txt; grep -h FINAL logs/${n}_s0.log >> logs/${n}_winner.txt
done
for n in KAUP3 KADN3; do
  WATCH="4,4;5,6" $P winner.py runs/${n}_s0_step300.pt runs/${n}_s0_step800.pt runs/${n}_s0_step2400.pt 2>&1 | grep -v Warn > logs/${n}_winner.txt; grep -h FINAL logs/${n}_s0.log >> logs/${n}_winner.txt
done
echo DONE
