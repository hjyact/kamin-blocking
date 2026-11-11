#!/bin/bash
# seeds 1,2 of the Sec.-6 invasion test (round116) and the takeover law (round115).
# predict: every rho <= 0.97 excluded (final gate <= 0.01), rho = 1 invades (gate >= 0.9), at both LM competences;
# takeover: LM-alone reaches ~100% later for larger rho, within +-50% of 350/(1-rho) (rho 0.5, 0.8, 0.9; 0.95 may not finish in 4000 steps)
cd .; P=~/casmi26/main/.venv/bin/python
for S in 1 2; do
 for PRE in 4000 600; do for R in 0.5 0.7 0.9 0.97 1.0; do
  REL=$R PRE=$PRE STEPS=$((PRE+4000)) SEED=$S EVERY=500 $P synth_tool3.py inv_p${PRE}_r${R}_s$S mix > logs/inv_p${PRE}_r${R}_s$S.log 2>&1
 done; done
 for R in 0.5 0.8 0.9 0.95; do REL=$R SEED=$S STEPS=4000 EVERY=250 $P synth_tool.py rel${R}_mix_s$S mix > logs/rel${R}_mix_s$S.log 2>&1; done
done
for S in 1 2; do
 for PRE in 4000 600; do for R in 0.5 0.7 0.9 0.97 1.0; do f=logs/inv_p${PRE}_r${R}_s$S.log
  echo "s$S PRE=$PRE REL=$R q@attach: $(grep "^GATE $PRE " $f | awk '{print $4}') final: $(grep FINAL $f | awk '{print $4}') gate_end: $(grep '^GATE' $f | tail -1 | awk '{print $3}')"
 done; done
 for R in 0.5 0.8 0.9 0.95; do echo "s$S rel=$R: $(grep 'LM-alone' logs/rel${R}_mix_s$S.log | awk '{printf "%s:%s ", $2, $NF}')"; done
done > logs/round204.txt
echo DONE
