"""Post-hoc fix on the released CPLM-135M: short fine-tune with (a) the original mixture loss only (control) or
(b) mixture loss + plain CE of the LM head (<copy> slot excluded). Then LM-alone / full eval split by seen/new tokens.
python cplm_fix.py SCRATCH {mix|fix} STEPS"""
import json, sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
S, mode, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]); repo = "nthngdy/cplm-135M"
tok = AutoTokenizer.from_pretrained(repo, revision="full")
m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
cid = m.config.copy_token_id; tr = json.load(open(f"{S}/train_texts.json")); va = json.load(open(f"{S}/val_texts.json"))
def enc(texts, T=1024):
    out = [tok(t, return_tensors="pt").input_ids[0, :T + 1] for t in texts]; L = min(x.size(0) for x in out)
    return torch.stack([x[:L] for x in out]).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.0); m.train(); m.config.inference_mode = "natural"; B, acc = 2, 4
for st in range(steps):
    opt.zero_grad(set_to_none=True)
    for a in range(acc):
        k = (st * acc + a) * B; ids = enc(tr[k:k + B]); x, y = ids[:, :-1], ids[:, 1:]
        with torch.autocast("cuda", torch.bfloat16):
            out = m.lm_model(input_ids=x, output_hidden_states=True, return_dict=True); h = out.hidden_states[-1]
            pc, sink = m._copy_dist_full(h, x, None)
        mixed = m._mix(out.logits, pc, sink)
        loss = F.nll_loss(mixed.reshape(-1, mixed.size(-1)), y.reshape(-1))
        if mode == "fix":
            lg = out.logits.float().clone(); lg[..., cid] = float("-inf")
            loss = loss + F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
        (loss / acc).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if st % 50 == 0: print(mode, st, round(loss.item(), 4), flush=True)
m.eval()
@torch.no_grad()
def ev(imode):
    m.config.inference_mode = imode; acc_ = {"all": [0.0, 0], "seen": [0.0, 0], "new": [0.0, 0]}
    for d in va:
        ids = tok(d, return_tensors="pt").input_ids[:, :1025].cuda()
        with torch.autocast("cuda", torch.bfloat16): lo = m(ids[:, :-1]).logits.float()
        t = ids[0, 1:]; nll = -lo.log_softmax(-1)[0].gather(-1, t.unsqueeze(-1)).squeeze(-1); inp = ids[0, :-1]; L = t.size(0)
        seen = ((inp.unsqueeze(0) == t.unsqueeze(1)) & torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool))).any(-1)
        for kk, msk in (("all", torch.ones_like(seen)), ("seen", seen), ("new", ~seen)): acc_[kk][0] += nll[msk].sum().item(); acc_[kk][1] += msk.sum().item()
    return {kk: round(v[0] / v[1], 4) for kk, v in acc_.items()}
print("RESULT", mode, steps, "natural", ev("natural"), "no_copy", ev("no_copy"), flush=True)
from huggingface_hub import hf_hub_download
rows = [json.loads(l)["text"] for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))]
for imode in ("natural", "no_copy"):
    m.config.inference_mode = imode; ok = 0
    with torch.no_grad():
        for t in rows:
            ctx, w = t.rsplit(" ", 1); c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + " " + w, return_tensors="pt").input_ids[0]; n = full.size(0) - c.size(0)
            ids = full.unsqueeze(0).cuda()
            with torch.autocast("cuda", torch.bfloat16): lo = m(ids[:, :-1]).logits.float()[0]
            ok += int((lo[-n:].argmax(-1) == ids[0, -n:]).all())
    print("LAMBADA", mode, steps, imode, round(ok / len(rows), 4), flush=True)
