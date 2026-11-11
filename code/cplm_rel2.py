"""Per-token copy reliability vs LM-head delegation across CPLM-1.7B training. For revision REV: CPLM gives c_y (copy
distribution's probability of the target, renormalised over source tokens) and the LM head's renormalised NLL; the vanilla
baseline at the same step gives its NLL. Saves per-token arrays; deletes the HF blobs after.  python cplm_rel.py REV"""
import json, sys, os, shutil, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
S = "."; rev = sys.argv[1]; SZ = sys.argv[2] if len(sys.argv) > 2 else "1.7B"
docs = json.load(open(f"{S}/val_texts.json"))[:120]; out = {}
tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision=rev)
toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
for repo in (f"nthngdy/cplm-{SZ}", f"nthngdy/cplm-baselines-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    C, LM = [], []
    with torch.no_grad():
        for ids in toks:
            ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]
            if "baselines" in repo:
                LM.append((-m(input_ids=x).logits.float()[0].log_softmax(-1).gather(-1, t[:, None]).squeeze(-1)).cpu())
            else:
                o = m.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); cid = m.config.copy_token_id
                p = o.logits.float()[0].softmax(-1); al = p[:, cid]
                LM.append((-(torch.log(p.gather(-1, t[:, None]).squeeze(-1) + 1e-30) - torch.log1p(-al))).cpu())
                pc, sink = m._copy_dist_full(o.hidden_states[-1], x, None); pc = pc.float()[0]
                C.append((pc.gather(-1, t[:, None]).squeeze(-1) / pc.sum(-1).clamp_min(1e-30)).cpu())
    key = "van" if "baselines" in repo else "cp"; out[key + "_nll"] = torch.cat(LM).numpy()
    if C: out["c"] = torch.cat(C).numpy()
    del m; torch.cuda.empty_cache()
    d = os.path.expanduser("~/tfresearch/hf6/hub/models--" + repo.replace("/", "--"))
    if "1.7B" in repo and os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
np.savez(f"{S}/rel_{SZ}_{rev}.npz", **out); print("saved", rev, {k: v.shape for k, v in out.items()}, flush=True)
