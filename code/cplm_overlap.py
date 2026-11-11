"""VCG anti-redundancy in released CPLM: on held-out tokens whose target occurs in the context, top-1 correctness of the copy
head, of CPLM's LM head (no <copy>) and of the vanilla LM. Prediction: CPLM's LM head overlaps the copy head less than a vanilla
LM trained without it does.  python cplm_overlap.py SZ"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; S = "."
tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full"); docs = json.load(open(f"{S}/val_texts.json"))[:120]
toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]; R = {}
for repo in (f"nthngdy/cplm-baselines-{SZ}", f"nthngdy/cplm-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); A, C, SEEN = [], [], []
    with torch.no_grad():
        for ids in toks:
            ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
            SEEN.append(((x[0][None, :] == t[:, None]) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1).cpu())
            if "baselines" in repo: A.append((m(input_ids=x).logits[0].argmax(-1) == t).cpu())
            else:
                o = m.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); lg = o.logits.float()[0]; lg[:, m.config.copy_token_id] = -1e9
                A.append((lg.argmax(-1) == t).cpu()); pc, _ = m._copy_dist_full(o.hidden_states[-1], x, None); C.append((pc.float()[0].argmax(-1) == t).cpu())
    R["van" if "baselines" in repo else "cp"] = torch.cat(A); R["seen"] = torch.cat(SEEN)
    if C: R["copy"] = torch.cat(C)
    del m; torch.cuda.empty_cache()
s = R["seen"]; c = R["copy"][s].float()
for k in ("cp", "van"):
    a = R[k][s].float(); both = (a * c).mean().item(); ind = a.mean().item() * c.mean().item()
    print(SZ, "LM=" + ("CPLM LM head" if k == "cp" else "vanilla LM  "), f"LM {a.mean():.3f} copy {c.mean():.3f} both {both:.3f} independent {ind:.3f} ratio {both / ind:.2f}  P(LM right | copy right) {both / c.mean():.3f}  P(LM right | copy wrong) {(a * (1 - c)).mean() / (1 - c).mean():.3f}", flush=True)
