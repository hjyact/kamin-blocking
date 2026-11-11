#!/bin/bash
# second seed of the stage-rule run of Fig. 3a (HI_PTR). predict: the head the pointer reads passes 0.05 before step 400 (plain seed-1 onset 1432)
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
$P train.py --name HI_PTR --seed 1 --tokens 100000000.0 --d 512 --layers 8 --heads 8 --seq 1024 --bs 32 --micro 8 --lr 0.003 --wd 0.1 --cheads 4 --ckeylayer -1 --crank 64 --train_file train.bin --hdrop 0.0 --eval_tokens 4000000.0 --nosave --lmaux 0.0 --creadmid -1 --vthink -1 --deleg_at -1.0 --mosk 0 --attnfloor 0.0 --oracle -1.0 --attnptr --oeps 0.01 --nosave > logs/HI_PTR_s1.log 2>&1
$P -c "
import json
L=[json.loads(l) for l in open('logs/HI_PTR_s1.log') if l.startswith('{') and '\"hi\"' in l]; prev=None; on=None
for r in L:
    h=r['hi'][1]
    if h>0.05 and on is None: on=prev[0]+(0.05-prev[1])*(r['step']-prev[0])/(h-prev[1]) if prev else r['step']
    prev=(r['step'],h)
print('pointer-read head onset', on, 'end', L[-1]['hi'])" > logs/round206.txt
echo DONE
