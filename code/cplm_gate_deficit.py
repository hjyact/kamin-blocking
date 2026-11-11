"""Replicator equilibrium of the learned gate: at a stationary gate, E[r_c | context] = lambda, with r_c = lambda c_y / p_y the
copy head's posterior responsibility. Bin held-out tokens by the gate (P(<copy>)) and compare mean r_c with mean gate.
python cplm_equil.py SZ"""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; S = "."; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open(f"{S}/val_texts.json"))[:120]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); VN = []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); VN.append((-V(input_ids=ids[:, :-1]).logits.float()[0].log_softmax(-1).gather(-1, ids[0, 1:, None]).squeeze(-1)).cpu())
del V; torch.cuda.empty_cache(); QN = []
m = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); G, RC = [], []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]
        o = m.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); P = o.logits.float()[0].softmax(-1); a = P[:, m.config.copy_token_id]
        q = P.gather(-1, t[:, None]).squeeze(-1) / (1 - a).clamp_min(1e-9)
        pc, sk = m._copy_dist_full(o.hidden_states[-1], x, None); pc = pc.float()[0]; sk = sk.float()[0]; pcy = pc.gather(-1, t[:, None]).squeeze(-1)
        lam = a * (1 - sk); p = (1 - lam) * q + a * pcy   # the model's own mixture: effective gate a(1 - a_sink), copy mass a * p_copy(y)
        G.append(lam.cpu()); QN.append((-torch.log(q.clamp_min(1e-30))).cpu()); RC.append((a * pcy / p.clamp_min(1e-30)).cpu())
g = torch.cat(G).numpy(); rc = torch.cat(RC).numpy(); d = (torch.cat(QN) - torch.cat(VN)).numpy(); edges = [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0001]
print(SZ, "tokens", len(g), "| mean gate", round(g.mean(), 4), "mean r_c", round(rc.mean(), 4))
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (g >= lo) & (g < hi)
    if s.sum() > 50: print(f"  gate [{lo:.2f},{hi:.2f}): n={s.sum():6d}  LM-head minus vanilla NLL {d[s].mean():+.3f}")
