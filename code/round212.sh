#!/bin/bash
# noise hypothesis for the head race: double the batch (64 sequences, 200M tokens: same 3051 steps and schedule, half the gradient noise).
# predict: blocked pair-2 run (pull L3H2 0-300, shared readout 300-800) leaves L6H1 above 0.10 at the end (bs 32: 0.021-0.070); control at bs 64 forms L6H1 or L3H2 > 0.2
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
B="--seed 0 --tokens 200e6 --bs 64 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 800,3050 --nosave"
$P train.py --name K2PRE64 $B --nudge 0.05 --nudge_spec "3,2,0,300" > logs/K2PRE64_s0.log 2>&1
$P train.py --name K2CTL64 $B > logs/K2CTL64_s0.log 2>&1
for n in K2PRE64 K2CTL64; do echo $n; WATCH="3,2;6,1" $P winner.py runs/${n}_s0_step800.pt runs/${n}_s0_step3050.pt 2>&1 | grep top3; done > logs/round212.txt
echo DONE
