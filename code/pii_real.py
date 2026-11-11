"""Blocking as memorisation control on realistic records (ai4privacy pii-masking-200k, English). 400 training and 80 held-out
records, each with one high-risk secret (email, password, card/account/phone number, IPv4). Fine-tune the released 135M LM
on the full record text. MODE plain | card (secret supplied in context, all record tokens trained) | cardall (every PII value
of the record supplied) | goldfish (every 4th token dropped from the loss). Attacks on the secret, given the record text before
it: greedy extraction, teacher-forced token accuracy, rank among 99 real values of the same type. Utility: held-out loss on
non-PII tokens.   python pii_real.py MODE STEPS"""
import sys, os, json, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
mode, steps = sys.argv[1], int(sys.argv[2]); SEED = int(os.environ.get("SEED", "0")); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained(os.environ.get("BASE", "nthngdy/cplm-baselines-135M"), revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
SECRET = {"EMAIL", "PASSWORD", "CREDITCARDNUMBER", "ACCOUNTNUMBER", "PHONENUMBER", "IPV4"}
rows = [json.loads(l) for l in open(hf_hub_download("ai4privacy/pii-masking-200k", "english_pii_43k.jsonl", repo_type="dataset"))]
recs, pool = [], {}
for r in rows:
    pm = r["privacy_mask"]; pm = json.loads(pm) if isinstance(pm, str) else pm
    sec = [e for e in pm if e["label"] in SECRET]
    for e in sec: pool.setdefault(e["label"], []).append(e["value"])
    if len(sec) == 1 and sec[0]["start"] >= 40 and len(r["source_text"]) < 600: recs.append((r["source_text"], sec[0], pm))
rr = random.Random(0); rr.shuffle(recs); train, held = recs[:400], recs[400:480]
def card(rec):
    text, sec, pm = rec
    if mode == "card": return f"[Record] {sec['value']}\n"
    if mode == "cardall": return "[Record] " + "; ".join(e["value"] for e in pm) + "\n"
    return ""
def pii_mask(text, pm, ids_offsets):   # True on tokens overlapping any PII span
    return [any(not (b <= e["start"] or a >= e["end"]) for e in pm) for a, b in ids_offsets]
def step(batch):
    tot = 0.0
    for rec in batch:
        text = rec[0]; pre = tok(card(rec)).input_ids if mode in ("card", "cardall") else []; d = tok(text).input_ids
        ids = torch.tensor([pre + d], device="cuda")
        with torch.autocast("cuda", torch.bfloat16): lg = m(input_ids=ids[:, :-1]).logits.float()[0]
        tgt = ids[0, 1:]; w = torch.zeros_like(tgt, dtype=torch.float); w[max(len(pre) - 1, 0):] = 1.0
        if mode == "goldfish": w[torch.arange(len(w), device="cuda") % 4 == 3] = 0.0
        l = (F.cross_entropy(lg, tgt, reduction="none") * w).sum() / w.sum()
        if mode == "dp":   # DP-SGD: per-example gradient clipped to norm C, summed in ACC; noise added in the step
            gs = torch.autograd.grad(l, PARAMS); nrm = torch.sqrt(sum((g.float() ** 2).sum() for g in gs)); f = min(1.0, CLIP / (nrm.item() + 1e-6))
            for a_, g in zip(ACC, gs): a_.add_(g, alpha=f)
        else: (l / len(batch)).backward()
        tot += l.item()
    if mode == "dp":
        for p_, a_ in zip(PARAMS, ACC): p_.grad = (a_ + torch.randn_like(a_) * SIGMA * CLIP) / len(batch); a_.zero_()
    return tot / len(batch)
@torch.no_grad()
def attacks(recs_, K=99):
    m.eval(); r = random.Random(7); ext = acc_n = acc_d = 0; ranks = []
    for text, sec, pm in recs_:
        prefix = text[:sec["start"]].rstrip(); val = text[sec["start"]:sec["end"]]; sp = " " if text[sec["start"] - 1] == " " else ""
        p = tok(prefix).input_ids; full = tok(prefix + sp + val).input_ids; x = torch.tensor([p], device="cuda")
        out = x
        for _ in range(len(full) - len(p) + 4): out = torch.cat([out, m(input_ids=out).logits[:, -1].argmax(-1, keepdim=True)], 1)
        ext += int(val in tok.decode(out[0, len(p):]))
        X = torch.tensor([full], device="cuda"); lg = m(input_ids=X[:, :-1]).logits.float()[0, len(p) - 1:]; acc_n += (lg.argmax(-1) == X[0, len(p):]).sum().item(); acc_d += len(full) - len(p)
        cands = [val] + r.sample([v for v in pool[sec["label"]] if v != val], K); seqs = [tok(prefix + sp + c).input_ids for c in cands]; L = max(map(len, seqs))
        S = torch.full((len(seqs), L), tok.eos_token_id, device="cuda"); M = torch.zeros(len(seqs), L - 1, device="cuda")
        for i, q in enumerate(seqs): S[i, :len(q)] = torch.tensor(q); M[i, len(p) - 1:len(q) - 1] = 1
        lp = (m(input_ids=S[:, :-1]).logits.float().log_softmax(-1).gather(-1, S[:, 1:, None]).squeeze(-1) * M).sum(-1)
        ranks.append(int((lp > lp[0]).sum().item()) + 1)
    m.train(); n = len(recs_); return dict(extract=round(ext / n, 3), tok_acc=round(acc_n / acc_d, 3), mean_rank=round(sum(ranks) / n, 1), top1=round(sum(x == 1 for x in ranks) / n, 3))
@torch.no_grad()
def utility():
    m.eval(); tot = cnt = 0
    for text, sec, pm in held:
        enc = tok(text, return_offsets_mapping=True); ids = enc.input_ids; mask = pii_mask(text, pm, enc.offset_mapping)
        x = torch.tensor([ids], device="cuda"); nll = F.cross_entropy(m(input_ids=x[:, :-1]).logits.float()[0], x[0, 1:], reduction="none")
        keep = torch.tensor([not k for k in mask[1:]], device="cuda"); tot += nll[keep].sum().item(); cnt += keep.sum().item()
    m.train(); return round(tot / cnt, 4)
print("records", len(recs), "START utility", utility(), flush=True)
PARAMS = [p_ for p_ in m.parameters() if p_.requires_grad]; CLIP = float(os.environ.get("CLIP", "1.0")); SIGMA = float(os.environ.get("SIGMA", "1.0"))
ACC = [torch.zeros_like(p_) for p_ in PARAMS] if mode == "dp" else []
brng = random.Random(SEED + 1000); opt = torch.optim.AdamW(m.parameters(), lr=float(os.environ.get("LR", "2e-4")), weight_decay=0.0); m.train()
for st in range(steps):
    opt.zero_grad(set_to_none=True); loss = step(brng.sample(train, 16)); opt.step()
    if st % 250 == 0: print(mode, st, round(loss, 4), flush=True)
print("RESULT", mode, steps, "train", attacks(train[:200]), "held", attacks(held), "utility", utility(), flush=True)
