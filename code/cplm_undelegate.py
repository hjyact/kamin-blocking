"""Training-free undelegation of a released CPLM's LM head (no copy head used): add -log(1 - lambda) to the logits of tokens
already in the context, lambda = P(<copy>) from the LM head itself (the sink is part of the copy head, so we use alpha), then
renormalise. Compare LM-alone NLL with and without the correction and the vanilla model, split by seen/new targets.
python cplm_undelegate.py SZ"""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:120]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
R = {k: [] for k in ("van", "lm", "fix", "fixh", "seen")}
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
        lv = V(input_ids=x).logits.float()[0].log_softmax(-1)
        lg = M.lm_model(input_ids=x).logits.float()[0]; a = lg.softmax(-1)[:, cid]; lg[:, cid] = -1e9
        inctx = torch.zeros_like(lg, dtype=torch.bool); tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        inctx.scatter_(1, torch.where(tri, x[0][None, :].expand(T, T), x[0][None, :1].expand(T, T)), tri)   # token ids seen at <= t (pad with first token, which is in context anyway)
        boost = -torch.log1p(-a.clamp(max=0.999))[:, None]
        nl = lambda L: -L.log_softmax(-1).gather(-1, t[:, None]).squeeze(-1)
        R["van"].append(-lv.gather(-1, t[:, None]).squeeze(-1).cpu()); R["lm"].append(nl(lg).cpu())
        R["fix"].append(nl(lg + boost * inctx).cpu()); R["fixh"].append(nl(lg + 0.5 * boost * inctx).cpu())
        R["seen"].append(inctx.gather(-1, t[:, None]).squeeze(-1).cpu())
R = {k: torch.cat(v).numpy() for k, v in R.items()}; s = R["seen"]
for k in ("van", "lm", "fix", "fixh"):
    print(SZ, f"{k:5s} all {R[k].mean():.4f}  seen {R[k][s].mean():.4f}  new {R[k][~s].mean():.4f}", flush=True)
