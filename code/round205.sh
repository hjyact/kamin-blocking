#!/bin/bash
# second seeds: plain 305M of Table 1, and the compute-optimal (77M, 1.5B tokens) trio behind the abstract's 2x claim.
# predict: XXL_BASE s1 within 0.01 of 4.028; CO_FIX full below CO_BASE by > 0.04; CO_COPY and CO_FIX full within 0.015;
# LM head alone (seensplit, one sample): CO_COPY above CO_BASE by > 0.10, CO_FIX above CO_BASE by < 0.06
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
$P train.py --name XXL_BASE --seed 1 --tokens 100e6 --d 1024 --layers 16 --heads 16 --micro 2 --lr 2e-3 --nosave > logs/XXL_BASE_s1.log 2>&1
C="--seed 1 --tokens 1.5e9 --train_file train_big.bin"
$P train.py --name CO_BASE $C > logs/CO_BASE_s1.log 2>&1
$P train.py --name CO_COPY $C --cachelm --cachesoft --cacheproj --cachefast > logs/CO_COPY_s1.log 2>&1
$P train.py --name CO_FIX $C --cachelm --cachesoft --cacheproj --cachefast --lmaux 1.0 > logs/CO_FIX_s1.log 2>&1
grep -h FINAL logs/XXL_BASE_s1.log logs/CO_*_s1.log > logs/round205.txt
$P seensplit.py runs/CO_BASE_s1.json runs/CO_COPY_s1.json runs/CO_FIX_s1.json runs/CO_BASE_s0.json runs/CO_COPY_s0.json runs/CO_FIX_s0.json 2>&1 | grep -v Warn >> logs/round205.txt
echo DONE
