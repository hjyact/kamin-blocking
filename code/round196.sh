#!/bin/bash
# prior-work control: Jacobs et al. (1991) own errors weighted by the gate. predict: LM-alone answer acc stays blocked (<10%) for jacobs and jacobsmix, three seeds; ours (fix) 76-100% as before
cd .; P=~/casmi26/main/.venv/bin/python
for S in 0 1 2; do for M in jacobs jacobsmix; do SEED=$S STEPS=4000 $P synth_tool.py ${M}_s$S $M > logs/${M}_s$S.log 2>&1; done; done
grep -h FINAL logs/jacobs_s*.log logs/jacobsmix_s*.log
echo DONE
