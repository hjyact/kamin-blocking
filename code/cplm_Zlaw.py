"""Context-level reweighting law, zero parameters: deficit_seen = -log(1-lam) + log Z, deficit_new = log Z, with
Z = (1-lam) Pc + (1 - Pc), lam = CPLM effective gate, Pc = vanilla probability mass on tokens already in the context.
Binned by gate, predicted vs actual deficit (CPLM LM head NLL - vanilla NLL).  python cplm_Zlaw.py SZ"""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:120]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
A, P, G, S = [], [], [], []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
        pv = V(input_ids=x).logits.float()[0].softmax(-1)
        o = M.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); lg = o.logits.float()[0]; a = lg.softmax(-1)[:, cid]; lg[:, cid] = -1e9
        _, sk = M._copy_dist_full(o.hidden_states[-1], x, None); lam = a * (1 - sk.float()[0])
        Vn = min(pv.size(-1), lg.size(-1)); inctx = torch.zeros(T, Vn, dtype=torch.bool, device=x.device)
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)); inctx.scatter_(1, torch.where(tri, x[0][None, :].expand(T, T), x[0][None, :1].expand(T, T)), tri)
        Pc = (pv[:, :Vn] * inctx).sum(-1); Z = (1 - lam) * Pc + (1 - Pc); seen = inctx.gather(1, t[:, None]).squeeze(-1)
        pred = torch.log(Z) - torch.where(seen, torch.log1p(-lam.clamp(max=0.999)), torch.zeros_like(lam))
        act = -lg.log_softmax(-1).gather(-1, t[:, None]).squeeze(-1) + torch.log(pv.gather(-1, t[:, None]).squeeze(-1).clamp_min(1e-30))
        A.append(act.cpu()); P.append(pred.cpu()); G.append(lam.cpu()); S.append(seen.cpu())
a, p, g, s = (torch.cat(v).numpy() for v in (A, P, G, S))
print(SZ, f"corr(pred, actual) seen {np.corrcoef(p[s], a[s])[0,1]:.3f}  all {np.corrcoef(p, a)[0,1]:.3f}")
for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
    b = (g >= lo) & (g < hi)
    print(f"  gate [{lo:.2f},{hi:.2f})  seen: pred {p[b & s].mean():+.3f} actual {a[b & s].mean():+.3f} -log(1-lam) {-np.log(1 - np.clip(g[b & s], 0, .999)).mean():+.3f} | new: pred {p[b & ~s].mean():+.3f} actual {a[b & ~s].mean():+.3f} (n={(b & ~s).sum()})")
