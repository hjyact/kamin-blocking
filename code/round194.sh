#!/bin/bash
# why do Adam-trained circuits still race? Adam's normaliser is set by a parameter's other signals. Identical cues, random-sign
# gradient of size 1 on every factor (5th field) as a proxy for the LM's other objectives.
# predict (later gate share at end, start 1/3 for r=2, 0.2 for r=4): Adam+noise p3 below start (amplified), p2 near start, p1 toward .5;
# Adam without noise (same run) erases for every p; SGD+noise p3 amplified as without noise.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
G=""
for o in "a:1" "a:0" "s:1"; do for p in 1 2 3; do for a in 0.02 0.04 0.08; do G="$G;$p:$a:0.02:$o"; done; done; done
G="${G#;}"
$P train.py --name GNOISE --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "4,4,300,3051;5,6,300,3051" --nudge 0.05 --nudge_spec "4,4,0,300;5,6,0,300" --gate_cfg "$G" --gate_lr 0.003 --gate_same --save_at 2400 --nosave > logs/GNOISE_s0.log 2>&1
echo DONE
