#!/bin/bash
# second seed of conditioned inhibition (Table 4). predict: product-trained head's index (in-context minus absent) below the plain seed-1 model's by > 0.2; with the extra term within 0.15 of plain; all indices > 0
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
$P train.py --name POEONLY --seed 1 --tokens 100000000.0 --d 512 --layers 8 --heads 8 --seq 1024 --bs 32 --micro 8 --lr 0.003 --wd 0.1 --cachelm --cachesoft --cacheproj --cheads 4 --ckeylayer -1 --crank 64 --eval_tokens 4000000.0 --lmaux 0.0 --cachepoe --poeonly --creadmid -1 --vthink -1 --deleg_at -1.0 --mosk 0 --attnfloor 0.0 > logs/POEONLY_s1.log 2>&1
$P train.py --name POEFIX --seed 1 --tokens 100000000.0 --d 512 --layers 8 --heads 8 --seq 1024 --bs 32 --micro 8 --lr 0.003 --wd 0.1 --cachelm --cachesoft --cacheproj --cheads 4 --ckeylayer -1 --crank 64 --eval_tokens 4000000.0 --lmaux 1.0 --cachepoe --poeonly --creadmid -1 --vthink -1 --deleg_at -1.0 --mosk 0 --attnfloor 0.0 > logs/POEFIX_s1.log 2>&1
$P inhibit.py runs/G0C0R0V0_s1.json runs/POEONLY_s1.json runs/POEFIX_s1.json 2>&1 | grep -v Warn > logs/round207.txt
echo DONE
