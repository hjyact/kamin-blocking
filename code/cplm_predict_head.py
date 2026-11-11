"""Zero-parameter prediction of the delegated LM head: q_pred(v) ∝ q_van(v) w(v), w = (1-lam) q_van / ((1-lam) q_van + lam c),
with CPLM's own effective gate lam and copy distribution c. Compare predicted and actual per-token deficit (CPLM LM head NLL
minus vanilla NLL).  python cplm_predict_head.py SZ"""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:40]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
A, P_, G = [], [], []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]
        qv = V(input_ids=x).logits.float()[0].softmax(-1)
        o = M.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); P = o.logits.float()[0].softmax(-1); a = P[:, cid]
        qc = P.clone(); qc[:, cid] = 0; qc = qc / qc.sum(-1, keepdim=True)
        pc, sk = M._copy_dist_full(o.hidden_states[-1], x, None); pc = pc.float()[0]; sk = sk.float()[0]
        lam = (a * (1 - sk))[:, None]; c = pc / (1 - sk)[:, None].clamp_min(1e-9)
        Vn = min(qv.size(-1), qc.size(-1)); qv, qc, c = qv[:, :Vn], qc[:, :Vn], c[:, :Vn]
        qp = qv
        for _k in range(int(__import__('os').environ.get('ITERS', '1'))):   # self-consistent weighted MLE: q = q_van * w(q) / Z
            w = (1 - lam) * qp / ((1 - lam) * qp + lam * c).clamp_min(1e-30); qp = qv * w; qp = qp / qp.sum(-1, keepdim=True)
        g = lambda d: -torch.log(d.gather(-1, t[:, None]).squeeze(-1).clamp_min(1e-30))
        A.append((g(qc) - g(qv)).cpu()); P_.append((g(qp) - g(qv)).cpu()); G.append(lam[:, 0].cpu())
a = torch.cat(A).numpy(); p = torch.cat(P_).numpy(); g = torch.cat(G).numpy()
print(SZ, f"tokens {len(a)}  mean actual deficit {a.mean():+.4f}  mean predicted {p.mean():+.4f}  corr {np.corrcoef(a, p)[0, 1]:.3f}  corr with -log(1-gate) {np.corrcoef(a, -np.log(1 - g + 1e-6))[0, 1]:.3f}")
o = np.argsort(p); 
for b in np.array_split(o, 8): print(f"  predicted {p[b].mean():+.3f}  actual {a[b].mean():+.3f}")
