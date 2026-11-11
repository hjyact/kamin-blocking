#!/bin/bash
# scale test at 608M (d1536, 16 layers), 100M tokens, one seed each: plain, copy head, copy head + extra term (lr 1.5e-3, chosen before the runs).
# predict: copy-head LM head alone worse than plain by > 0.10; with the extra term within 0.05 of plain; both copy models' full loss better than plain by > 0.1
# all three evaluated (LM head alone, seen/new) with seensplit.py, full mixture with cpalone.py, on one validation sample; the plain checkpoint is deleted after its evaluation (disk)
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
S="--seed 0 --tokens 100e6 --d 1536 --layers 16 --heads 16 --micro 1 --lr 1.5e-3"
$P train.py --name XXXL_BASE $S > logs/XXXL_BASE_s0.log 2>&1
$P seensplit.py runs/XXXL_BASE_s0.json 2>&1 | grep -v Warn > logs/round201.txt
[ -s logs/round201.txt ] && rm -f runs/XXXL_BASE_s0.pt
$P train.py --name XXXL_COPY $S --cachelm --cachesoft --cacheproj --cachefast > logs/XXXL_COPY_s0.log 2>&1
$P train.py --name XXXL_AUX1 $S --cachelm --cachesoft --cacheproj --cachefast --lmaux 1.0 > logs/XXXL_AUX1_s0.log 2>&1
$P seensplit.py runs/XXXL_COPY_s0.json runs/XXXL_AUX1_s0.json 2>&1 | grep -v Warn >> logs/round201.txt
$P cpalone.py runs/XXXL_COPY_s0.json runs/XXXL_AUX1_s0.json 2>&1 | grep -v Warn >> logs/round201.txt
grep -h FINAL logs/XXXL_*_s0.log >> logs/round201.txt
echo DONE
