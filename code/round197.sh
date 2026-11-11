#!/bin/bash
# seeds for the synthetic Table-4 cells (main text). predict seed 0 reproduces the table; seeds 1,2: overexp both members lose confidence (<.5 from ~.7);
# supercond margin rise within 25% of m; backward blocking (kalman, plus) negative on S; overshadowing LM and helper each .3-.5 with overlap <.1
cd .; P=~/casmi26/main/.venv/bin/python
for S in 0 1 2; do
  echo "== seed $S" 
  SEED=$S $P synth_overexp.py product 2>&1 | tail -n 4
  for M in 1 2 3; do SEED=$S $P synth_supercond.py $M 2>&1 | tail -n 2; done
  SEED=$S $P synth_bb.py kalman plus 2>&1 | tail -n 3
  SEED=$S $P synth_shap.py plain 2>&1 | tail -n 4
done > logs/round197.txt
echo DONE
