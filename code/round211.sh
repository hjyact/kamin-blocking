#!/bin/bash
# the race mechanism on heads, isolated: the late head keeps LM-loss gradient only at induction-copyable positions (its induction credit), none elsewhere (competing objectives).
# predict: late head at step 2000 > 0.15 in both pairs (controls K2LONG/KALONG: 0.010 / 0.014; fully cut off: 0.015 / 0.013)
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
C="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --nudge 0.05 --save_at 2000 --nosave --solo_copy"
$P train.py --name K2SCOPY $C --oread_spec "3,2,300,2000;6,1,300,2000" --nudge_spec "3,2,0,300" --solo_spec "6,1,300,2000" > logs/K2SCOPY_s0.log 2>&1
$P train.py --name KASCOPY $C --oread_spec "4,4,300,2000;5,6,300,2000" --nudge_spec "4,4,0,300" --solo_spec "5,6,300,2000" > logs/KASCOPY_s0.log 2>&1
{ echo K2SCOPY; WATCH="3,2;6,1" $P winner.py runs/K2SCOPY_s0_step2000.pt 2>&1 | grep top3; echo KASCOPY; WATCH="4,4;5,6" $P winner.py runs/KASCOPY_s0_step2000.pt 2>&1 | grep top3; } > logs/round211.txt
echo DONE
