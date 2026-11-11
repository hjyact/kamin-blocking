#!/bin/bash
# blocking probability vs number of demonstrations K (6 seeds each). predict: fraction blocked (zero-shot <= 0.65) rises with K; K=1 <= 3/6, K=8 >= 4/6
cd .; P=python
for K in 1 2 4 8; do for S in 10 11 12 13 14 15; do echo "K=$K $(K=$K SEED=$S $P demo_block.py demo 300 2>&1 | grep FINAL)"; done; done > kcurve.txt
echo DONE >> kcurve.txt
