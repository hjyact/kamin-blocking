"""Delegation test on the released Copy-Paste LMs: LM head alone (inference_mode no_copy) vs the full model vs the vanilla
baseline trained with the same recipe. Per-token NLL on FineWeb-edu text, split by whether the target already occurred."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
S = sys.argv[1]; size = sys.argv[2]; docs = json.load(open(f"{S}/val_texts.json"))
def run(repo, mode=None, T=1024):
    tok = AutoTokenizer.from_pretrained(repo, revision="full")
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    if mode: m.config.inference_mode = mode
    acc = {"all": [0.0, 0], "seen": [0.0, 0], "new": [0.0, 0]}
    with torch.no_grad():
        for d in docs:
            ids = tok(d, return_tensors="pt").input_ids[:, :T + 1].cuda()
            if ids.size(1) < 64: continue
            out = m(ids[:, :-1]).logits.float(); tgt = ids[:, 1:]
            nll = -out.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
            inp = ids[0, :-1]; t = tgt[0]; L = t.size(0)
            seen = ((inp.unsqueeze(0) == t.unsqueeze(1)) & torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool))).any(-1)
            for k, msk in (("all", torch.ones_like(seen)), ("seen", seen), ("new", ~seen)): acc[k][0] += nll[msk].sum().item(); acc[k][1] += msk.sum().item()
    r = {k: round(v[0] / v[1], 4) for k, v in acc.items()}; r["frac_seen"] = round(acc["seen"][1] / acc["all"][1], 3)
    del m; torch.cuda.empty_cache(); return r
if "--skipbase" not in sys.argv: print("baseline", run(f"nthngdy/cplm-baselines-{size}"), flush=True)
print("cplm natural", run(f"nthngdy/cplm-{size}", "natural"), flush=True)
print("cplm no_copy", run(f"nthngdy/cplm-{size}", "no_copy"), flush=True)
