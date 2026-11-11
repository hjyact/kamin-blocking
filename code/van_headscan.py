"""Vanilla baseline: zero each head; drop in the LM head's mean probability of the target on the repeated half of a random
sequence (and on seen targets of held-out text).  python van_headscan.py SZ"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:16]; R = torch.cat([tok(d, return_tensors="pt").input_ids[:, :513] for d in docs if len(tok(d).input_ids) >= 513][:8]).cuda()
H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
m = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
nh = m.config.num_attention_heads; hd = m.config.hidden_size // nh; NL = len(m.model.layers)
def pt(Z, rep):
    x, t = Z[:, :-1], Z[:, 1:]; T = x.size(1)
    with torch.no_grad(): p = m(input_ids=x).logits.float().softmax(-1).gather(-1, t[..., None]).squeeze(-1)
    if rep: return p[:, H:].mean().item()
    sn = ((x.unsqueeze(1) == t.unsqueeze(2)) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1); return p[sn].mean().item()
base = {"text": pt(R, False), "repeat": pt(X, True)}; res = {"text": {}, "repeat": {}}
for L in range(NL):
    for h in range(nh):
        def pre(mod, args, hh=h):
            x = args[0].clone(); x[..., hh * hd:(hh + 1) * hd] = 0; return (x,)
        k = m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
        res["text"][(L, h)] = base["text"] - pt(R, False); res["repeat"][(L, h)] = base["repeat"] - pt(X, True); k.remove()
for nm in ("text", "repeat"):
    d = sorted(res[nm].items(), key=lambda kv: -kv[1]); tot = sum(max(v, 0) for _, v in d)
    print(SZ, "vanilla", nm, f"base p(target) {base[nm]:.3f}; top:", [(f"L{L}h{h}", round(v, 3)) for (L, h), v in d[:6]], f"| top-1 share {d[0][1] / tot:.2f}, top-5 {sum(v for _, v in d[:5]) / tot:.2f}", flush=True)
