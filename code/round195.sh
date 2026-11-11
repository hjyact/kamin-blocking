#!/bin/bash
# second seed of the two depth runs shown in the main text (fig 3d, Sec. 10). predict same ordering: GSAME p2 r2 share 0.333 +- .005, p3 r2 < 0.30, p1 r2 > 0.43, Adam r2 > 0.38
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
G=""
for p in 1 2 3; do for a in 0.02 0.03 0.04 0.08; do G="$G;$p:$a:0.02:s"; done; done
G="${G#;};3:0.04:0.02:a;3:0.08:0.02:a;3:0.02:0.02:a"
C="--seed 1 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 4,4,300,3051;5,6,300,3051 --nudge 0.05 --nudge_spec 4,4,0,300;5,6,0,300 --gate_lr 0.003 --nosave"
$P train.py --name GSAME $C --gate_cfg "$G" --gate_same > logs/GSAME_s1.log 2>&1
$P train.py --name GDEPTH3 $C --gate_cfg "$G" > logs/GDEPTH3_s1.log 2>&1
echo DONE
