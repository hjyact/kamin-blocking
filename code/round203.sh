#!/bin/bash
# second seed of the Sec.-10 clock numbers. baseline seed 1 = CLKNULLa_s1. predict: onset at lr x0.5 and x2 within 25% of the baseline (a learning-rate clock gives x2 / x0.5);
# the 1000-step block over 700-1700 delays the onset more than the one over 0-1000.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
B="--seed 1 --tokens 100e6 --oracle 1.0 --nosave"
$P train.py --name LRP1.5e-3 $B --orc_windows 9000:9001 --lr 1.5e-3 > logs/LRP1.5e-3_s1.log 2>&1
$P train.py --name LRP6e-3 $B --orc_windows 9000:9001 --lr 6e-3 > logs/LRP6e-3_s1.log 2>&1
$P train.py --name CLKW0 $B --orc_windows 0:1000 > logs/CLKW0_s1.log 2>&1
$P train.py --name CLKW700 $B --orc_windows 700:1700 > logs/CLKW700_s1.log 2>&1
$P -c "exec(open('onset.py').read().split('for n in sys.argv')[0])
for n in ['CLKNULLa','LRP1.5e-3','LRP6e-3','CLKW0','CLKW700']: print(n, onset(f'logs/{n}_s1.log'))" > logs/round203.txt
echo DONE
