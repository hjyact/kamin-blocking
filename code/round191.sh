#!/bin/bash
# depth law in isolation: two induction heads pulled together (0-300), then 14 independent gate pairs read their DETACHED outputs
# (g_A zA + g_B zB, shared CE, only gates learn); gate = product of p scalars. Same features, same loss, only parametrisation differs.
# predict (B share gB/(gA+gB) at step 1000, head start ratio r = gA0/gB0 in 1,1.5,2,4):
#  p=1 SGD: head start washes out (share spread over r < 0.05); p=2: share ordered by r and close to 1/(1+r) early;
#  p=3 SGD: head start amplified (share at r=2 below half of share at r=1); p=3 Adam: washes out like p=1.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
G=""
for p in 1 2 3; do for a in 0.02 0.03 0.04 0.08; do G="$G;$p:$a:0.02:s"; done; done
G="${G#;};3:0.04:0.02:a;3:0.08:0.02:a;3:0.02:0.02:a"
$P train.py --name GDEPTH --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "4,4,300,3051;5,6,300,3051" --nudge 0.05 --nudge_spec "4,4,0,300;5,6,0,300" --gate_cfg "$G" --gate_lr 0.1 --save_at 2400 --nosave > logs/GDEPTH_s0.log 2>&1
echo DONE
