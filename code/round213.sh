#!/bin/bash
# saturation hypothesis: bounded gates g = M*sigmoid(prod-5) (M 0.35 / 0.5) let one gate fit the target alone. identical cues, head start 2:1 (start share 0.333).
# predict: bounded Adam gates race (later gate's end share < 0.2), against erasure for the unbounded Adam control (about 0.47).
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
$P train.py --name GSAT --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "4,4,300,3051;5,6,300,3051" --nudge 0.05 --nudge_spec "4,4,0,300;5,6,0,300" --gate_cfg "1:0.04:0.02:a:0:0;1:0.04:0.02:a:0:0:0.35;2:0.04:0.02:a:0:0:0.35;3:0.04:0.02:a:0:0:0.35;1:0.04:0.02:s:0:0:0.35;2:0.04:0.02:s:0:0:0.35;1:0.04:0.02:a:0:0:0.5;2:0.04:0.02:a:0:0:0.5" --gate_lr 0.003 --gate_same --nosave > logs/GSAT_s0.log 2>&1
echo DONE
