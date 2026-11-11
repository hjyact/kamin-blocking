#!/bin/bash
# second seed of the two 305M main-text claims. predict: XH_ABV final head >= 1.5x XH_BLW; XDLG1200 val below XDLG0 by > 0.007
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
S="--d 1024 --layers 16 --heads 16 --micro 2 --lr 2e-3"
C="--seed 1 --tokens 100e6 $S --cachelm --cachesoft --cacheproj --cachefast --nosave"
$P train.py --name XDLG0 $C > logs/XDLG0_s1.log 2>&1
$P train.py --name XDLG1200 $C --deleg_from 1200 > logs/XDLG1200_s1.log 2>&1
$P train.py --name XH_ABV --seed 1 --tokens 100e6 $S --oracle 0.5 --orc_from 1477 --nosave > logs/XH_ABV_s1.log 2>&1
$P train.py --name XH_BLW --seed 1 --tokens 100e6 $S --oracle 0.5 --nosave > logs/XH_BLW_s1.log 2>&1
for n in XDLG0 XDLG1200 XH_ABV XH_BLW; do grep '"step": 3050' logs/${n}_s1.log | cut -c1-200 | sed "s/^/$n /"; done > logs/round200.txt
echo DONE
