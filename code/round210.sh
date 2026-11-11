#!/bin/bash
# the race mechanism on real heads: the late head B gets no LM-loss gradient (--solo_spec), only the shared readout's. head start for A (pull 0-300), readout 300-2000.
# predict: with B solo its induction score at step 2000 > 0.15 (not excluded); control (B also trained by the LM loss) < 0.10, in both pairs.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
C="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --nudge 0.05 --save_at 2000 --nosave"
$P train.py --name K2LONG $C --oread_spec "3,2,300,2000;6,1,300,2000" --nudge_spec "3,2,0,300" > logs/K2LONG_s0.log 2>&1
$P train.py --name K2SOLO $C --oread_spec "3,2,300,2000;6,1,300,2000" --nudge_spec "3,2,0,300" --solo_spec "6,1,300,2000" > logs/K2SOLO_s0.log 2>&1
$P train.py --name KALONG $C --oread_spec "4,4,300,2000;5,6,300,2000" --nudge_spec "4,4,0,300" > logs/KALONG_s0.log 2>&1
$P train.py --name KASOLO $C --oread_spec "4,4,300,2000;5,6,300,2000" --nudge_spec "4,4,0,300" --solo_spec "5,6,300,2000" > logs/KASOLO_s0.log 2>&1
for n in K2LONG K2SOLO; do echo $n; WATCH="3,2;6,1" $P winner.py runs/${n}_s0_step2000.pt 2>&1 | grep top3; done > logs/round210.txt
for n in KALONG KASOLO; do echo $n; WATCH="4,4;5,6" $P winner.py runs/${n}_s0_step2000.pt 2>&1 | grep top3; done >> logs/round210.txt
echo DONE
