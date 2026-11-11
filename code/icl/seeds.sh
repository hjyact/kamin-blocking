#!/bin/bash
# seeds 1,2 of the demonstration-blocking result. predict: demo zero-shot <= 0.65, plain >= 0.90, fix >= 0.80
cd .; P=python
for S in 1 2; do for md in plain demo fix; do SEED=$S $P demo_block.py $md 300 2>&1 | grep FINAL; done; done > seeds.txt
echo DONE >> seeds.txt
