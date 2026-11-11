"""Neural-cache helper (Grave et al. 2017) attached to the released plain baseline LM (cplm-baselines-135M, full).
p = (1-lam) p_LM + lam p_cache, p_cache(w) = sum_{j<t, x_{j+1}=w} softmax_j(theta cos(h_t, h_j)); lam = sigmoid(linear(h_t)) learned gate, theta learned scalar, h
detached (the helper is competent without training). MODE plain | mix | fix (mix + plain CE). Eval: LM alone, seen/new.
python cplm_cache.py SCRATCH MODE STEPS"""
import json, sys, os, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
S, mode, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]); torch.manual_seed(int(os.environ.get("SEED", "0")))
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained("nthngdy/cplm-baselines-135M", revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
tr = json.load(open(f"{S}/train_texts.json")); va = json.load(open(f"{S}/val_texts.json"))
gate = torch.nn.Linear(576, 1).cuda(); torch.nn.init.zeros_(gate.weight); torch.nn.init.constant_(gate.bias, math.log(0.2 / 0.8)); glam = gate.bias; lth = torch.nn.Parameter(torch.tensor(math.log(10.0), device="cuda"))
def cache_lp(h, x, y):   # log p_cache(y_t) for every t, from keys h_j (j < t) with values y_j
    hn = F.normalize(h.detach().float(), dim=-1); T = hn.size(1); sc = lth.exp() * (hn @ hn.transpose(1, 2))
    past = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1); sc = sc.masked_fill(~past, -1e9)
    la = sc - sc.logsumexp(-1, keepdim=True); match = (y.unsqueeze(2) == y.unsqueeze(1)) & past
    lpc = la.masked_fill(~match, -1e9).logsumexp(-1); return torch.where(match.any(-1), lpc, torch.full_like(lpc, -1e9))
@torch.no_grad()
def ev():
    m.eval(); acc_ = {"all": [0.0, 0], "seen": [0.0, 0], "new": [0.0, 0]}; mixl = [0.0, 0]
    for d in va:
        ids = tok(d, return_tensors="pt").input_ids[:, :1025].cuda(); x, y = ids[:, :-1], ids[:, 1:]
        with torch.autocast("cuda", torch.bfloat16): o = m(input_ids=x, output_hidden_states=True, return_dict=True)
        lp = o.logits.float().log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1); nll = -lp[0]
        lam = torch.sigmoid(gate(o.hidden_states[-1].detach().float())).squeeze(-1); mx = torch.logaddexp(torch.log1p(-lam) + lp, torch.log(lam) + cache_lp(o.hidden_states[-1], x, y))
        mixl[0] += -mx.sum().item(); mixl[1] += mx.numel(); t = y[0]; inp = x[0]; L = t.size(0)
        seen = ((inp.unsqueeze(0) == t.unsqueeze(1)) & torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool))).any(-1)
        for kk, msk in (("all", torch.ones_like(seen)), ("seen", seen), ("new", ~seen)): acc_[kk][0] += nll[msk].sum().item(); acc_[kk][1] += msk.sum().item()
    m.train(); return {kk: round(v[0] / v[1], 4) for kk, v in acc_.items()}, round(mixl[0] / mixl[1], 4)
print("START LM", ev(), flush=True)
def enc(texts, T=1024):
    out = [tok(t, return_tensors="pt").input_ids[0, :T + 1] for t in texts]; L = min(x.size(0) for x in out)
    return torch.stack([x[:L] for x in out]).cuda()
opt = torch.optim.AdamW([{"params": list(m.parameters()), "lr": float(os.environ.get("LR", "1e-4"))}, {"params": list(gate.parameters()) + [lth], "lr": 1e-3}], weight_decay=0.0); m.train(); acc = 8
for st in range(steps):
    opt.zero_grad(set_to_none=True)
    for a in range(acc):
        k = (st * acc + a) % len(tr); ids = enc(tr[k:k + 1]); x, y = ids[:, :-1], ids[:, 1:]
        with torch.autocast("cuda", torch.bfloat16): o = m(input_ids=x, output_hidden_states=True, return_dict=True)
        lp = o.logits.float().log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        if mode == "plain": loss = -lp.mean()
        else:
            lam = torch.sigmoid(gate(o.hidden_states[-1].detach().float())).squeeze(-1); mx = torch.logaddexp(torch.log1p(-lam) + lp, torch.log(lam) + cache_lp(o.hidden_states[-1], x, y))
            loss = -mx.mean() - (lp.mean() if mode == "fix" else 0.0)
        (loss / acc).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if st % 100 == 0: print(mode, st, round(loss.item(), 4), "lam", round(lam.mean().item() if mode != "plain" else 0.0, 3), "theta", round(lth.exp().item(), 2), flush=True)
print("RESULT", mode, steps, "LM", ev(), flush=True)
