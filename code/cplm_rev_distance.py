"""Blocking only takes what the LM could learn: sliding-window (256) CPLM-1.7B vs its vanilla sw baseline. For held-out tokens
whose target occurred earlier, bin by distance to the previous occurrence; report vanilla NLL, CPLM LM-head NLL and the deficit.
python cplm_sw_distance.py"""
import json, os, glob, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys; REV = sys.argv[1] if len(sys.argv) > 1 else "sw"
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-1.7B", revision="full")
docs = json.load(open("val_texts.json")); toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]; toks = [t for t in toks if t.size(1) >= 900][:150]
R = {}
for repo in ("nthngdy/cplm-baselines-1.7B", "nthngdy/cplm-1.7B"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision=REV, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); NL, DI = [], []
    with torch.no_grad():
        for ids in toks:
            ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
            if "baselines" in repo: lg = m(input_ids=x).logits.float()[0]
            else:
                lg = m.lm_model(input_ids=x).logits.float()[0]; lg[:, m.config.copy_token_id] = -1e9
            NL.append((-lg.log_softmax(-1).gather(-1, t[:, None]).squeeze(-1)).cpu())
            eq = (x[0][None, :] == t[:, None]) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            last = torch.where(eq, torch.arange(T, device=x.device)[None, :], torch.full_like(eq, -1, dtype=torch.long)).amax(-1)
            DI.append(torch.where(last >= 0, torch.arange(T, device=x.device) - last, torch.full_like(last, -1)).cpu())
    R["van" if "baselines" in repo else "cp"] = torch.cat(NL).numpy(); R["d"] = torch.cat(DI).numpy()
    del m; torch.cuda.empty_cache()
    for f in glob.glob(snapshot_download(repo, revision=REV, local_files_only=True) + "/*.safetensors"): os.remove(os.path.realpath(f))
d = R["d"]
for lo, hi in ((-1, 0), (0, 16), (16, 64), (64, 256), (256, 512), (512, 1100)):
    s = (d >= lo) & (d < hi) if lo >= 0 else (d < 0)
    nm = "new (not in context)" if lo < 0 else f"distance [{lo},{hi})"
    print(f"{nm:22s} n={s.sum():6d}  vanilla {R['van'][s].mean():.3f}  CPLM LM head {R['cp'][s].mean():.3f}  deficit {(R['cp'][s] - R['van'][s]).mean():+.3f}", flush=True)
