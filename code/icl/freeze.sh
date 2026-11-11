#!/bin/bash
# causal test of the freeze mechanism: demo training with attention frozen (predict zero-shot >= 0.9) or MLPs frozen (predict <= 0.65), K=2, seeds 10-12
cd .; P=python
for F in attn mlp; do for S in 10 11 12; do echo "$F $(K=2 FREEZE=$F SEED=$S $P demo_block.py demo 300 2>&1 | grep -E 'trainable|FINAL' | tr '\n' ' ')"; done; done > freeze.txt
echo DONE >> freeze.txt
