"""Which backbone heads does the released CPLM copy head read? Zero each head in turn; drop in the copy head's mean
probability of the target on seen-target positions, on real held-out text and on a repeated random sequence.
python cplm_headscan.py SZ"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:16]; R = torch.cat([tok(d, return_tensors="pt").input_ids[:, :513] for d in docs if len(tok(d).input_ids) >= 513][:8]).cuda()
H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
m = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); bb = m.lm_model
nh = bb.config.num_attention_heads; hd = bb.config.hidden_size // nh; NL = len(bb.model.layers)
def copyp(Z):
    x, t = Z[:, :-1], Z[:, 1:]; T = x.size(1)
    with torch.no_grad():
        o = bb(input_ids=x, output_hidden_states=True, return_dict=True); pc, _ = m._copy_dist_full(o.hidden_states[-1], x, None)
    sn = ((x.unsqueeze(1) == t.unsqueeze(2)) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1)
    return pc.float().gather(-1, t[..., None]).squeeze(-1)[sn].mean().item()
base = {"text": copyp(R), "repeat": copyp(X)}; res = {"text": {}, "repeat": {}}
for L in range(NL):
    for h in range(nh):
        def pre(mod, args, hh=h):
            x = args[0].clone(); x[..., hh * hd:(hh + 1) * hd] = 0; return (x,)
        k = bb.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
        for nm, Z in (("text", R), ("repeat", X)): res[nm][(L, h)] = base[nm] - copyp(Z)
        k.remove()
for nm in ("text", "repeat"):
    d = sorted(res[nm].items(), key=lambda kv: -kv[1]); tot = sum(max(v, 0) for _, v in d)
    print(SZ, nm, f"base copy p(target) {base[nm]:.3f}; top heads by drop:", [(f"L{L}h{h}", round(v, 3)) for (L, h), v in d[:6]], f"| top-1 share of positive drops {d[0][1] / tot:.2f}, top-5 {sum(v for _, v in d[:5]) / tot:.2f}", flush=True)
