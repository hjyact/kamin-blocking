# Code for "Language Models Trained With Helpers Stop Learning, and Forget, What the Helpers Know"

All paths are relative to this folder. `prep.py` / `prep_big.py` tokenise FineWeb-edu into `data/` (300M / 1.6B GPT-2 tokens).
Released models are loaded from the Hugging Face hub (`nthngdy/cplm-*`, `nthngdy/cplm-baselines-*`).

| Paper item | Script and command |
|---|---|
| Scaling table, fix table, compute multiplier | `train.py --tokens 100e6 [--d 768 --layers 12 --heads 12] --cachelm --cachesoft --cacheproj --cachefast [--lmaux 1.0]`; LM alone: `lmonly.py runs/X.json` |
| Seen/new split | `copyprobe.py runs/X.json` |
| Compute-optimal (77M, 1.5B tokens) | `train.py --tokens 1.5e9 --train_file train_big.bin ...` (plain / `--lmaux 1.0` / none / `--nodeleg`) |
| Simple alternatives | `train.py --init runs/EXPBASE_s0.pt --freeze_base ...` (two-stage); `--hdrop 0.5` (helper dropout) |
| Erasure in a 77M LM | `train.py --init runs/EXPBASE_s0.pt --tokens 100e6 [copy flags] [--lmaux 1.0]`, then `copyprobe.py` |
| Price decomposition | `train.py ... --pricelog`, then `price.py runs/X_s0_price.npz` |
| Released CPLMs: delegation, downstream | `cplm_eval.py`, `cplm_big.py`, `lambada.py`, `hellaswag.py` |
| Repairing released models | `cplm_fix.py SCRATCH {mix,fix} STEPS` |
| Graded blocking (135M/360M/1.7B) | `cplm_rel.py sw_42B|sw_126B|sw_200B`, `cplm_rel2.py full 135M|360M` |
| Released 135M + frozen 360M expert | `cplm_expert.py . {plain,mix,fix} 1000` (`OFF=8000` for the second slice) |
| Weak helper (neural cache) | `cplm_cache.py . {plain,mix,fix} 1000` |
| Retrieval blocking in released LMs | `rag_real.py {rag,fix,drop} STEPS` (`BASE=...360M`, `LORA=16 BASE=...1.7B`, `DISTRACT=2`) |
| Calculator, partial calculator | `synth_tool.py NAME {alone,mix,fix}`, `synth_tool2.py` |
| Erasure, invasion phase diagram, erasure floor | `synth_tool3.py NAME mix` with `PRE=4000 STEPS=10000`, `REL=`, `EPS2=`, `LR2=`, `DISJ=1 [DISJMIX=1]` |
| Takeover law | `synth_tool.py NAME {mix,sub}` with `REL=` |
| Retrieval blocking (synthetic), dropout | `synth_rag.py NAME {rag,fix,alone}` with `PRE=2000`, `DROP=` |
| Theorem 1 / corollary / optimiser | `synth_vocab.py K {mix,alone} STEPS` with `SGD=1 GB=3.0`, `FIXW=`, `COV=0.5`, `MUON=1` |
| RL reading | `synth_tool_rl.py` |
| Secrets (appendix) | `pii.py`, `pii_real.py` |
| Figures | `figs.py`, `fig_blocking.py`, `fig_theory.py` |

Seeds are set with `--seed` (train.py) or `SEED=` (synthetic scripts).
- synth_alpha.py (App. alpha): SGD=1 GB=3 python synth_alpha.py 64 {-1,0,1,sum} 20000; FIXW=1 for the fix; EPS=1e-2/1e-4 for the rate law
- synth_bb.py / synth_bb2.py (App. bb): python synth_bb.py {sgd,diag,kalman} {plus,minus}; python synth_bb2.py {sgd,adam} {plus,minus}
- round122.sh / train.py --oracle f --oeps 1e-2 [--lmaux 1] (App. orc): oracle product helper on induction-copyable positions
- circuits.py (App. orc): round123.sh trains saved f=0/f=1 runs; python circuits.py runs/ORCS0_s0.json runs/ORCS1.0_s0.json
- cplm_circ.py (Table qkov): HF_HOME=... python cplm_circ.py {135M,360M,1.7B}
- round124.sh / train.py --attnptr [--lmaux 1] (App. orc): pointer copy head from the backbone's last-layer head 0
- round125.sh (App. orc, credit conservation): HI_PLAIN / HI_PTR with per-step head induction logging (rec 'hi')
- PTRSCAF / PTRAUX (App. scaf): train.py --attnptr --ptr_until 0.2 ; train.py --ptraux 1.0 --ptr_until 0.2
- PTRSCAFFIX / PTRAUXRC (App. scaf): --attnptr --ptr_until 0.2 --lmaux 1 ; --ptraux 1 --ptrrc --ptr_until 0.2
- cplm_circ_traj.py (App. qkov trajectory): python cplm_circ_traj.py nthngdy/cplm-1.7B full_42B full_84B full_126B full_168B full_200B
- inhibit.py (App. products of experts): python inhibit.py runs/ORCS0_s0.json runs/POEONLY_s0.json runs/POEFIX_s0.json runs/CACHEPROJ_s0.json
- synth_shap.py (App. shap): python synth_shap.py {plain,fixLM,BM,shapley}
- SHAP2 (App. shap): train.py --cachelm --cachesoft --cacheproj --cachefast --lmaux 1 --cpaux 1 ; python cpalone.py runs/AUX1_s0.json runs/SHAP2_s0.json
- cplm_lens.py / cplm_lens2.py (App. qkov, logit lens): python cplm_lens.py {135M,360M}; python cplm_lens2.py {135M,360M}
- cplm_stagefix.py (App. qkov repair): BS=4 python cplm_stagefix.py {last 2,first 10,last 10,all 0} 800
- round126.sh (App. orc, dose-response): train.py --attnptr --ptrlam {0.2,0.8}
- ptrfull.py (App. orc): python ptrfull.py runs/ORCS0_s0.json runs/PTR_s0.json runs/PTRFIX_s0.json
- synth_tool3.py mask (Table partial, Prop. complement): SEED={0,1,2} python synth_tool3.py MASK mask
- cplm_overlap.py (App. shap, VCG anti-redundancy): python cplm_overlap.py {135M,360M}
- MASKSEEN (App. block, Prop. complement bound): train.py --maskseen ; python seensplit.py runs/ORCS0_s0.json runs/CACHEPROJ_s0.json runs/AUX1_s0.json runs/MASKSEEN_s0.json
- CSTOPFIX (App. block): train.py --cachelm --cachesoft --cacheproj --cachefast --cstopgrad --lmaux 1 ; python seensplit.py runs/CSTOPFIX_s0.json
- NEWW2 (App. block): train.py --newweight 2 ; python seensplit.py runs/NEWW2_s0.json
- synth_vocab.py late escape (App. vocab, Prop. sign): GB=3 COV=0.5 EVERY=2000 python synth_vocab.py 64 mix 30000 ; add SGD=1 for the control
- synth_vocab_grad.py (App. vocab, gradient dominance): GB=3 COV=0.5 EVERY=2000 GEVERY=2000 python synth_vocab_grad.py 64 mix 30000
- synth_vocab_split.py (responsibility-split Adam): GEVERY=1000000000 GB=3 COV=0.5 EVERY=5000 SPLIT=1 python synth_vocab_split.py 64 mix 30000
- round127.sh (App. vocab, split Adam, negative): train.py --splitadam [--splitrand]
- synth_overexp.py (App. overexp): python synth_overexp.py {product,mixture,product_own}
- synth_supercond.py (App. overexp): python synth_supercond.py {1,2,3} ; python synth_supercond.py 2 own
- synth_alpha.py fixed gate (App. alpha, failed escape-time prediction): SGD=1 FIXLAM=0.9 EVERY=250 python synth_alpha.py 64 {-1,-0.5,0,0.5} 40000
- cplm_equil.py (Table equil): python cplm_equil.py {135M,360M,1.7B}
- cplm_gate_deficit.py (App. equil): python cplm_gate_deficit.py {135M,360M}
- cplm_gate_deficit2.py (App. equil, seen/new split): python cplm_gate_deficit2.py {135M,360M}
- cplm_predict_head.py (App. equil, zero-parameter head prediction): python cplm_predict_head.py {135M,360M}
- cplm_undelegate.py (App. equil, training-free repair): python cplm_undelegate.py {135M,360M}
- lambada_undel.py (App. equil): python lambada_undel.py {135M,360M}
- cplm_equil_rev.py (App. equil, trajectory): python cplm_equil_rev.py 1.7B full_{42B,84B,126B,168B}
- cplm_sw_distance.py (App. orc, sliding-window): python cplm_sw_distance.py
- cplm_sw_gate.py (App. orc, sw responsibility by distance): python cplm_sw_gate.py (copy p column has a normalisation bug; gate and responsibility are correct)
- cplm_rev_distance.py (App. orc, hybrid): python cplm_rev_distance.py hybrid
- cplm_circ_traj2.py (SFT erasure, sliding window): HALF=128 python cplm_circ_traj2.py nthngdy/cplm-1.7B sw sw_sft ; same for nthngdy/cplm-baselines-1.7B
- lambada_errors.py (App. equil, substitution errors): python lambada_errors.py {135M,360M}
- round128.sh / round129.sh (App. shap, helper-helper in a real LM): train.py --cachefast ... --expert screen/base_s0.json [--lmaux 1] --seq 512 ; SEQ1=513 python cpalone.py ...
- round130.sh (App. shap, overlapping expert): --expert runs/ORCS0_s0.json --lmaux 1
- cplm_Zlaw.py (App. equil, context-level renormalisation law): python cplm_Zlaw.py {135M,360M}
- cplm_residual.py (App. equil, residual law): python cplm_residual.py {135M,360M}
- cplm_ablate.py (App. mech, causal ablation): python cplm_ablate.py {135M,360M}
- ablate77.py (App. mech): python ablate77.py runs/ORCS0_s0.json runs/CACHEPROJ_s0.json runs/AUX1_s0.json
- cplm_ablate_text.py (App. mech): python cplm_ablate_text.py {135M,360M}
- cplm_headscan.py (App. mech): python cplm_headscan.py 135M
- van_headscan.py (App. mech): python van_headscan.py 135M
- PTRL3 (App. orc, pointer at layer 3): train.py --attnptr --ptrlayer 3 ; python ablate77.py runs/PTRL3_s0.json
- round131.sh (App. orc, pointer at layers 1 and 0): train.py --attnptr --ptrlayer {1,0}
- PTRL2 (Table ptrlayer): train.py --attnptr --ptrlayer 2
- ptr_resp.py (Table ptrlayer, responsibility by position): python ptr_resp.py runs/PTRL{0,1,2}_s0.json runs/PTR_s0.json
- round132.sh (Table ptrlayer, responsibility logged during training): train.py --attnptr --ptrlayer {1,0} (OSTAT now logs responsibility on copyable/other positions)
- ORCH0.3 (App. orc, threshold test): train.py --oracle 0.3 --nosave
- round133.sh (App. orc, onset law): train.py --oracle {0.15,0.5} --nosave
- ORCH1.0FIX (App. orc): train.py --oracle 1.0 --lmaux 1.0 --nosave
- INDW2 (App. orc): train.py --indweight 2 --nosave
- round134.sh, ORCH1.0FIX seed 1, ORCH1.0FIXe4 (App. orc, open discrepancy): --indweight 2 seeds 1,2; --oracle 1 --lmaux 1 --seed 1; --oracle 1 --oeps 1e-4 --lmaux 1

## Credit windows, hysteresis and delegation order (appendix G, "When credit is removed matters"; appendix "Delegating after the onset")
```
# onset noise floor (same seed, oracle code path, window never active)
python train.py --name CLKNULLa --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nosave
# f=1 credit windows (steps a:b, comma-separated for several)
python train.py --name CLKW2 --seed 0 --tokens 100e6 --oracle 1.0 --orc_from 700 --orc_until 1700 --nosave
python train.py --name CLKK300 --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 300:600 --nosave
# blocking after the onset (upkeep curve): f in 0.5 0.75 0.9 1.0
python train.py --name CLKABV0.75 --seed 0 --tokens 100e6 --oracle 0.75 --orc_from 1600 --nosave
# drop false-copy (1) or as many random (2) positions during the block
python train.py --name CLKFM1 --seed 0 --tokens 100e6 --oracle 1.0 --orc_from 1600 --falsemask 1 --nosave
# delegation order with the cache helper: from step t_d, or state-triggered
C="--seed 0 --tokens 100e6 --cachelm --cachesoft --cacheproj --cachefast --nosave"
python train.py --name DLG1200 $C --deleg_from 1200
python train.py --name DLGAUTO $C --autodeleg 0.05
# credit ledger
python train.py --name LEDGER --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --save_at 800,1200,1600,2400 --nosave
python ledger.py runs/LEDGER_s0_step1600.pt; python ledger_sign.py runs/LEDGER_s0_step1600.pt
python onset.py CLKNULLa CLKW2   # onsets from logs/<name>_s0.log
```
```
# null slot (per-head unnormalised null key, value 0), its blocking run, and null mass by position class
python train.py --name NULLPLAIN --seed 0 --tokens 100e6 --nullslot --oracle 1.0 --orc_windows 9000:9001 --save_at 1600 --nosave
python train.py --name NULLFRZ --seed 0 --tokens 100e6 --nullslot --oracle 1.0 --orc_from 1393 --nosave
LEDGER_ARGS="--nullslot" python ledger.py runs/NULLPLAIN_s0_step1600.pt; python nullmass.py runs/NULLPLAIN_s0_step1600.pt
# drop new-target positions from the loss early / late
python train.py --name UNS_EARLY --seed 0 --tokens 100e6 --oracle 0.0 --orc_windows 0:1000 --falsemask 3 --nosave
python train.py --name UNS_LATE --seed 0 --tokens 100e6 --oracle 0.0 --orc_windows 1200:4000 --falsemask 3 --nosave
```
```
# functional ledger (natural-text induction score), shortcut probe, per-class loss
NAT=1 python sledger.py runs/LEDGER_s0_step800.pt; python shortcut.py runs/LEDGER_s0_step800.pt; python classloss.py runs/LEDGER_s0_step800.pt
# early removal (steps 0-1000): --falsemask 1 false-copy, 3 new-target, 4 target-seen/current-new, 5 false-copy+new-target, 6 random (--dropfrac), 7 top-10 targets
python train.py --name LG_SEEN --seed 0 --tokens 100e6 --oracle 0.0 --orc_windows 0:1000 --falsemask 4 --nosave
python train.py --name LG_FREQ10 --seed 0 --tokens 100e6 --oracle 0.0 --orc_windows 0:1000 --falsemask 7 --nosave   # needs data/freq_order.npy (argsort of train token counts, first 50M tokens)
```
```
# learning-rate clocks: plain at x0.5/x2/x0.25 and with f=0.5 blocking
python train.py --name LRP1.5e-3 --seed 0 --tokens 100e6 --lr 1.5e-3 --oracle 1.0 --orc_windows 9000:9001 --nosave
python train.py --name LRF7.5e-4 --seed 0 --tokens 100e6 --lr 7.5e-4 --oracle 0.5 --nosave
```
```
# steering which head becomes the induction head: early pull on one head (steps < 300), then per-head scores
python train.py --name NUDGE44 --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nudge 0.05 --nudge_lh 4,4 --nudge_until 300 --save_at 300,1200,2400 --nosave
WATCH="4,4" python winner.py runs/NUDGE44_s0_step300.pt runs/NUDGE44_s0_step2400.pt
```
```
# Kamin design between heads (--nudge_spec "L,H,from,until;...")
python train.py --name KSEQ --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nudge 0.05 --nudge_spec "4,4,0,300;5,6,300,600" --save_at 300,600,1200,2400 --nosave
python train.py --name KCTL --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nudge 0.05 --nudge_spec "5,6,300,600" --save_at 300,600,1200,2400 --nosave
```
```
# Kamin blocking between heads under a shared error: pull one head first (attention, 0-300), then shared readout of both (--oread, 300-800)
C="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 4,4,300,800;5,6,300,800 --nudge 0.05 --save_at 300,800,2400 --nosave"
python train.py --name KBPRE $C --nudge_spec "5,6,0,300"; python train.py --name KAPRE $C --nudge_spec "4,4,0,300"
python train.py --name KOCTL --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "4,4,500,1000;5,6,500,1000" --save_at 500,1000,2400 --nosave
```
```
# second pair (L3H2, L6H1): control compound, C first, D first
B="--seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec 3,2,300,800;6,1,300,800 --save_at 300,800,2400 --nosave"
python train.py --name K2CTL $B; python train.py --name K2CPRE $B --nudge 0.05 --nudge_spec "3,2,0,300"; python train.py --name K2DPRE $B --nudge 0.05 --nudge_spec "6,1,0,300"
```
```
# release from blocking: place L4H4, zero its output from step 1200, track the other heads
python train.py --name ABL44 --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nudge 0.05 --nudge_spec "4,4,0,300" --ablate 4,4,1200 --save_at 1200,1350,1500,1800,2100,2400,3050 --nosave
python others.py runs/ABL44_s0_step1350.pt@4,4
```
```
# dose-response of blocking: pull C for 0-k steps (k = 50, 100, 150) before the shared readout
python train.py --name K2C100 --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "3,2,300,800;6,1,300,800" --save_at 300,2400 --nosave --nudge 0.05 --nudge_spec "3,2,0,100"
```
```
# pre-exposure: pull L3H2 to the previous position (--nudge_prev) before the shared-readout race
python train.py --name KLI --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --oread 0.1 --oread_spec "3,2,300,800;6,1,300,800" --nudge 0.05 --nudge_spec "3,2,0,300" --nudge_prev --save_at 300,800,2400 --nosave
HEAD=3,2 python prevscore.py runs/KLI_s0_step300.pt
```
python train.py --name KLI5 ... --nudge_prev --nudge_off 5   # useless-pattern control for pre-exposure; HEAD=3,2 OFF=5 python prevscore.py ...
python train.py --name KLI5D ... --nudge_detach   # pull trains only the head own q/k projections (no gradient into the residual)
python train.py --name K2CPREOWN ... --oread_own 1.0   # own-error readout per head in the compound phase (circuit-level Bush-Mosteller)
python train.py --name ABLTMP --seed 0 --tokens 100e6 --oracle 1.0 --orc_windows 9000:9001 --nudge 0.05 --nudge_spec "4,4,0,300" --ablate 4,4,1200,1800 --save_at 1200,1800,2400,3050 --nosave   # temporary removal

## Circuit-depth and race experiments (App. depth)
- Gate depth, distinct features: `round192.sh` (`--gate_cfg "p:gA:gB:s|a" --gate_lr 0.003`), second seed `round195.sh`; analysis `gamp.py logs/GDEPTH3_s0.log`
- Gate depth, identical cues: `round193.sh` (`--gate_same`); analysis `gshare.py logs/GSAME_s0.log`
- Adam with random signals: `round194.sh` (5th gate field = noise size); with a constant competing pull: `round208.sh`, `round209.sh` (6th field = pull)
- Bounded gates: `round213.sh` (7th field = bound M)
- Late head cut off from the LM loss: `round210.sh` (`--solo_spec`), copyable positions only: `round211.sh` (`--solo_copy`)
- Halved gradient noise: `round212.sh`; no weight decay: `round214.sh`
- Unblocking by outcome shifts: `round188.sh`, `round189.sh`, `round198.sh` (`--oread_T`, `--oread_rate`); head strength `strength.py`
- 608M scale test: `round201.sh`, `round202.sh`; compute-optimal second seed: `round205.sh`; seeds for Sec. 6, 10, Table 4: `round203.sh`, `round204.sh`, `round206.sh`, `round207.sh`
