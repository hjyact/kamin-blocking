"""Sliding-window CPLM-1.7B: per distance-to-previous-occurrence bin, mean effective gate, copy head's probability of the target
and its responsibility.  python cplm_sw_gate.py"""
import json, os, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-1.7B", revision="full")
docs = json.load(open("val_texts.json")); toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]; toks = [t for t in toks if t.size(1) >= 900][:150]
m = AutoModelForCausalLM.from_pretrained("nthngdy/cplm-1.7B", revision="sw", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = m.config.copy_token_id
G, C, RC, DI = [], [], [], []
with torch.no_grad():
    for ids in toks:
        ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
        o = m.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); P = o.logits.float()[0].softmax(-1); a = P[:, cid]
        q = P.gather(-1, t[:, None]).squeeze(-1) / (1 - a).clamp_min(1e-9)
        pc, sk = m._copy_dist_full(o.hidden_states[-1], x, None); pc = pc.float()[0]; sk = sk.float()[0]; pcy = pc.gather(-1, t[:, None]).squeeze(-1)
        lam = a * (1 - sk); p = (1 - lam) * q + a * pcy
        G.append(lam.cpu()); C.append((pcy / (1 - sk).clamp_min(1e-9)).cpu()); RC.append((a * pcy / p.clamp_min(1e-30)).cpu())
        eq = (x[0][None, :] == t[:, None]) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        last = torch.where(eq, torch.arange(T, device=x.device)[None, :], torch.full_like(eq, -1, dtype=torch.long)).amax(-1)
        DI.append(torch.where(last >= 0, torch.arange(T, device=x.device) - last, torch.full_like(last, -1)).cpu())
g, c, rc, d = (torch.cat(v).numpy() for v in (G, C, RC, DI))
for lo, hi in ((-1, 0), (0, 16), (16, 64), (64, 256), (256, 512), (512, 1100)):
    s = (d >= lo) & (d < hi) if lo >= 0 else (d < 0)
    print(f"{('new' if lo < 0 else f'[{lo},{hi})'):12s} n={s.sum():6d}  gate {g[s].mean():.3f}  copy p(target) {c[s].mean():.3f}  responsibility {rc[s].mean():.3f}", flush=True)
for f in glob.glob(snapshot_download("nthngdy/cplm-1.7B", revision="sw", local_files_only=True) + "/*.safetensors"): os.remove(os.path.realpath(f))
