"""CPLM: per layer (logit lens), probability of <copy> on the repeated half vs the first half, and the target's rank among
real tokens.  python cplm_lens2.py SZ"""
import sys, torch
from transformers import AutoModelForCausalLM
SZ = sys.argv[1]; H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
m = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); bb = m.lm_model; cid = m.config.copy_token_id
with torch.no_grad():
    o = bb(input_ids=X, output_hidden_states=True, return_dict=True); t = X[:, H + 1:]; pc2, pc1, rk = [], [], []
    for h in o.hidden_states[1:]:
        lg = bb.lm_head(bb.model.norm(h)).float(); P = lg.softmax(-1)
        pc2.append(P[:, H:-1, cid].mean().item()); pc1.append(P[:, 1:H - 1, cid].mean().item())
        l2 = lg[:, H:-1].clone(); l2[..., cid] = -1e9; tgt = l2.gather(-1, t[..., None]); rk.append((l2 > tgt).sum(-1).float().median().item())
print(SZ, "P(<copy>) repeated half", [round(v, 2) for v in pc2]); print(SZ, "P(<copy>) first half   ", [round(v, 2) for v in pc1]); print(SZ, "median target rank      ", [int(v) for v in rk])
