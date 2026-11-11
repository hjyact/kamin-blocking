#!/bin/bash
# 360M: predict K=2 demo blocked (zero-shot <= 0.65) in >= 5/6 seeds; fix >= 0.85; plain >= 0.90
cd .; P=python; export BASE=HuggingFaceTB/SmolLM-360M
for S in 10 11 12 13 14 15; do echo "demo $(K=2 SEED=$S $P demo_block.py demo 300 2>&1 | grep FINAL)"; done > scale360.txt
for S in 10 11; do echo "fix $(K=2 SEED=$S $P demo_block.py fix 300 2>&1 | grep FINAL)"; echo "plain $(K=2 SEED=$S $P demo_block.py plain 300 2>&1 | grep FINAL)"; done >> scale360.txt
echo DONE >> scale360.txt
