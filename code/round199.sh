#!/bin/bash
# rerun of the seed-2 latent-inhibition runs of round198 (disk was full); same predictions
# (2) latent inhibition, seed 2 with its own control: predict useless pattern (offset 5) -> L3H2 within 0.08 of the control; previous-token role -> L3H2 < 0.10 if the control's L3H2 > 0.2.
cd .; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; P=~/casmi26/main/.venv/bin/python
B="--tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 300,800,2400 --nosave"
$P train.py --name K2UP3 --seed 1 $B --nudge 0.05 --nudge_spec "3,2,0,300" --oread_T 3 > logs/K2UP3_s1.log 2>&1
B="--tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 800,2400 --nosave"
$P train.py --name K2CTL --seed 2 $B > logs/K2CTL_s2.log 2>&1
$P train.py --name KLI --seed 2 $B --nudge 0.05 --nudge_spec "3,2,0,300" --nudge_prev > logs/KLI_s2.log 2>&1
$P train.py --name KLI5 --seed 2 $B --nudge 0.05 --nudge_spec "3,2,0,300" --nudge_prev --nudge_off 5 > logs/KLI5_s2.log 2>&1
for r in K2CTL_s2 KLI_s2 KLI5_s2; do
  echo $r >> logs/round199.txt
  WATCH="3,2;6,1" $P winner.py runs/${r}_step800.pt runs/${r}_step2400.pt 2>&1 | grep top3 >> logs/round199.txt
done
echo DONE
