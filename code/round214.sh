#!/bin/bash
# weight-decay hypothesis for the head race: AdamW's decoupled decay is a consistent pull toward zero on every parameter.
# predict: with --wd 0 the blocked pair-2 late head (L6H1) ends above 0.10 (wd 0.1: 0.021-0.070); control without head start forms a head > 0.2.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
B="--seed 0 --tokens 100e6 --wd 0 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 800,3050 --nosave"
$P train.py --name K2PREWD0 $B --nudge 0.05 --nudge_spec "3,2,0,300" > logs/K2PREWD0_s0.log 2>&1
$P train.py --name K2CTLWD0 $B > logs/K2CTLWD0_s0.log 2>&1
for n in K2PREWD0 K2CTLWD0; do echo $n; WATCH="3,2;6,1" $P winner.py runs/${n}_s0_step800.pt runs/${n}_s0_step3050.pt 2>&1 | grep top3; done > logs/round214.txt
echo DONE
