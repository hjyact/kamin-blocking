"""Residual law, zero parameters: the delegated LM head is what is left of a calibrated full distribution once the helper's
share is removed, q(v) = max(q_van(v) - alpha p_copy(v), eps) / Z. Per-token predicted vs actual deficit, binned.
python cplm_residual.py SZ"""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:60]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
A, P, S = [], [], []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
        pv = V(input_ids=x).logits.float()[0].softmax(-1)
        o = M.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); lg = o.logits.float()[0]; a = lg.softmax(-1)[:, cid]; lg[:, cid] = -1e9
        pc, sk = M._copy_dist_full(o.hidden_states[-1], x, None); pc = pc.float()[0]
        Vn = min(pv.size(-1), lg.size(-1), pc.size(-1)); pv, pc, lg = pv[:, :Vn], pc[:, :Vn], lg[:, :Vn]
        qp = (pv - a[:, None] * pc).clamp_min(1e-6); qp = qp / qp.sum(-1, keepdim=True)
        g = lambda d: d.gather(-1, t[:, None]).squeeze(-1)
        A.append((-g(lg.log_softmax(-1)) + torch.log(g(pv))).cpu()); P.append((-torch.log(g(qp)) + torch.log(g(pv))).cpu())
        S.append(((x[0][None, :] == t[:, None]) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1).cpu())
a, p, s = (torch.cat(v).numpy() for v in (A, P, S))
print(SZ, f"mean actual {a.mean():+.4f} predicted {p.mean():+.4f}  corr all {np.corrcoef(a, p)[0,1]:.3f} seen {np.corrcoef(a[s], p[s])[0,1]:.3f} new {np.corrcoef(a[~s], p[~s])[0,1]:.3f}")
o = np.argsort(p)
for b in np.array_split(o, 10): print(f"  predicted {p[b].mean():+.3f}  actual {a[b].mean():+.3f}")
