#!/bin/bash
# time course of the in-context vs in-weights race at K=2 (seeds 10-15), every 25 steps on 200 test items
cd .; P=python
for S in 10 11 12 13 14 15; do K=2 SEED=$S EVERY=25 NEV=200 $P demo_block.py demo 300 2>&1 | grep -E "^[0-9]+ |FINAL" | sed "s/^/s$S /"; done > race.txt
echo DONE >> race.txt
