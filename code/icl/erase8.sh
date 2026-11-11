#!/bin/bash
# erasure frequency: 100 steps plain then 300 with K=2 demos, seeds 13-20. predict >= 2/8 erased (>=0.9 at step 100, <=0.65 at the end)
cd .; P=python
for S in 13 14 15 16 17 18 19 20; do echo "s$S $(K=2 PRE=100 SEED=$S EVERY=100 NEV=200 $P demo_block.py erase 400 2>&1 | grep -E '^100 |FINAL' | awk '{print $4, $NF}' | tr '\n' ' ')"; done > erase8.txt
echo DONE >> erase8.txt
