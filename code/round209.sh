#!/bin/bash
# seed 1 repeat of round208; same predictions
# Adam race hypothesis: a consistent competing pull c on every gate factor (6th field) turns Adam's erasure into a threshold race.
# identical cues, head start 2:1 (start share 0.333). predict: Adam p=2,3 with c>0 (at some c) end share < 0.15, with c=0 >= 0.38;
# Adam p=1 with c: no race (share not below 0.30); at equal c, Adam's share below SGD's (p=2).
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
$P train.py --name GPULL --seed 1 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "4,4,300,3051;5,6,300,3051" --nudge 0.05 --nudge_spec "4,4,0,300;5,6,0,300" --gate_cfg "1:0.04:0.02:a:0:0;2:0.04:0.02:a:0:0;3:0.04:0.02:a:0:0;2:0.04:0.02:s:0:0;1:0.04:0.02:a:0:0.01;2:0.04:0.02:a:0:0.01;3:0.04:0.02:a:0:0.01;2:0.04:0.02:s:0:0.01;1:0.04:0.02:a:0:0.03;2:0.04:0.02:a:0:0.03;3:0.04:0.02:a:0:0.03;2:0.04:0.02:s:0:0.03;1:0.04:0.02:a:0:0.1;2:0.04:0.02:a:0:0.1;3:0.04:0.02:a:0:0.1;2:0.04:0.02:s:0:0.1" --gate_lr 0.003 --gate_same --nosave > logs/GPULL_s1.log 2>&1
echo DONE
