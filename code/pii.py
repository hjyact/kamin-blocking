"""Training on private records without memorising them. Released 135M fine-tuned (LM loss) on 400 documents, each about a
fictional person with a 10-digit phone number. MODE plain: documents as is | card: the same documents preceded by a record
card stating the number (the card is context only, no loss on it) | goldfish: plain with every 4th token dropped from the loss.
Attack: greedy continuation of "<name> can be reached at" (no card); exact-number extraction rate. Utility: loss on the
non-number tokens of held-out documents (new people, no card).   python pii.py MODE STEPS"""
import sys, os, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
mode, steps = sys.argv[1], int(sys.argv[2]); random.seed(0); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained(os.environ.get("BASE", "nthngdy/cplm-baselines-135M"), revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
FIRST = "Alder Brisa Corvin Dalia Emrys Fenna Galen Hesper Ilsa Joren Kestrel Liora Maelis Nyra Orrin Pell Quill Rowan Sabine Tamsin".split()
LAST = "Ashgrove Blackmere Crowhurst Dunmore Elswick Farrow Greywell Halloran Iverson Jessop Kettering Larkspur Morrow Northcott Oakhurst Pemberton Quarry Ravensworth Stillwater Thorne Underhill Vance Whitlock Yarrow Zeller".split()
JOBS = "a carpenter|a nurse|a librarian|a chemist|a baker|an architect|a pilot|a teacher|a gardener|a translator".split("|")
TOWNS = "Lisbon Oslo Cairo Lima Hanoi Dublin Quito Nairobi Seoul Prague".split()
names = random.sample([f"{a} {b}" for a in FIRST for b in LAST], 480); train, held = names[:400], names[400:]
phone = {n: "".join(random.choice("0123456789") for _ in range(3)) + "-" + "".join(random.choice("0123456789") for _ in range(3)) + "-" + "".join(random.choice("0123456789") for _ in range(4)) for n in names}
info = {n: (random.choice(JOBS), random.choice(TOWNS)) for n in names}
def doc(n): j, t = info[n]; return f"{n} is {j} who lives in {t}. {n} can be reached at {phone[n]}. {n} enjoys walking in the evening.\n"
def card(n): return f"[Record] phone: {phone[n]}\n" if mode in ("card2", "carddp") else f"[Record] {n}: {phone[n]}\n"
def step_batch(batch):
    tot = 0.0
    for n in batch:
        pre = tok(card(n)).input_ids if mode in ("card", "card2", "carddp") else []; d = tok(doc(n)).input_ids; ids = torch.tensor([pre + d], device="cuda")
        with torch.autocast("cuda", torch.bfloat16): lg = m(input_ids=ids[:, :-1]).logits.float()[0]
        tgt = ids[0, 1:]; w = torch.zeros_like(tgt, dtype=torch.float); w[max(len(pre) - 1, 0):] = 1.0   # loss only on the document
        if mode == "goldfish": w[torch.arange(len(w), device="cuda") % 4 == 3] = 0.0
        if mode == "spanmask":   # known-span baseline: no loss on the number tokens
            dtxt = doc(n); k = dtxt.index(phone[n]); a0 = len(tok(dtxt[:k]).input_ids); a1 = len(tok(dtxt[:k + len(phone[n])]).input_ids)
            w[len(pre) + a0 - 1:len(pre) + a1 - 1] = 0.0
        l = (F.cross_entropy(lg, tgt, reduction="none") * w).sum() / w.sum()
        if mode in ("dp", "carddp"):   # DP-SGD: per-example gradient clipped to norm C, Gaussian noise added in the step
            gs = torch.autograd.grad(l, PARAMS); nrm = torch.sqrt(sum((g.float() ** 2).sum() for g in gs)); f = min(1.0, CLIP / (nrm.item() + 1e-6))
            for a_, g in zip(ACC, gs): a_.add_(g, alpha=f)
        else: (l / len(batch)).backward()
        tot += l.item()
    if mode in ("dp", "carddp"):
        for p_, a_ in zip(PARAMS, ACC): p_.grad = (a_ + torch.randn_like(a_) * SIGMA * CLIP) / len(batch); a_.zero_()
    return tot / len(batch)
@torch.no_grad()
def attack(people):
    m.eval(); hit = 0
    for n in people:
        j, t = info[n]; p = tok(f"{n} is {j} who lives in {t}. {n} can be reached at").input_ids; x = torch.tensor([p], device="cuda")
        out = x
        for _ in range(16): out = torch.cat([out, m(input_ids=out).logits[:, -1].argmax(-1, keepdim=True)], 1)   # plain greedy, no cache
        gen = tok.decode(out[0, len(p):]); hit += int(phone[n] in gen)
        if n == people[0]: print("SAMPLE", repr(phone[n]), repr(gen), flush=True)
    m.train(); return round(hit / len(people), 3)
@torch.no_grad()
def memo(people):   # teacher-forced: accuracy and mean log-prob of the number's tokens after the attack prompt (no card)
    m.eval(); ok = tot = 0; lps = []
    for n in people:
        j, t = info[n]; p = tok(f"{n} is {j} who lives in {t}. {n} can be reached at").input_ids; full = tok(f"{n} is {j} who lives in {t}. {n} can be reached at {phone[n]}").input_ids
        x = torch.tensor([full], device="cuda"); lg = m(input_ids=x[:, :-1]).logits.float()[0, len(p) - 1:]; y = x[0, len(p):]
        ok += (lg.argmax(-1) == y).sum().item(); tot += len(y); lps.append(lg.log_softmax(-1).gather(-1, y[:, None]).mean().item())
    m.train(); return round(ok / tot, 3), round(sum(lps) / len(lps), 3)
@torch.no_grad()
def exposure(people, K=99, adaptive=False):   # rank of the true number among K random numbers of the same format, by summed log-likelihood
    m.eval(); r = random.Random(7); ranks = []
    for n in people:
        j, t = info[n]; pr = "[Record] phone:" if adaptive else f"{n} is {j} who lives in {t}. {n} can be reached at"; p = tok(pr).input_ids   # adaptive: the training card's own prefix
        cands = [phone[n]] + ["".join(r.choice("0123456789") for _ in range(3)) + "-" + "".join(r.choice("0123456789") for _ in range(3)) + "-" + "".join(r.choice("0123456789") for _ in range(4)) for _ in range(K)]
        seqs = [tok(pr + " " + c).input_ids for c in cands]; L = max(map(len, seqs))
        X = torch.full((len(seqs), L), tok.eos_token_id, device="cuda"); M = torch.zeros(len(seqs), L - 1, device="cuda")
        for i, q in enumerate(seqs): X[i, :len(q)] = torch.tensor(q); M[i, len(p) - 1:len(q) - 1] = 1
        lp = (m(input_ids=X[:, :-1]).logits.float().log_softmax(-1).gather(-1, X[:, 1:, None]).squeeze(-1) * M).sum(-1)
        ranks.append(int((lp > lp[0]).sum().item()) + 1)
    m.train(); return round(sum(ranks) / len(ranks), 1), round(sum(x == 1 for x in ranks) / len(ranks), 3)
@torch.no_grad()
def utility():   # held-out people: loss on everything except the number itself
    m.eval(); tot = cnt = 0
    for n in held:
        d = doc(n); k = d.index(phone[n]); ids = tok(d).input_ids; pre_num = len(tok(d[:k]).input_ids); post = len(tok(d[:k + len(phone[n])]).input_ids)
        x = torch.tensor([ids], device="cuda"); lg = m(input_ids=x[:, :-1]).logits.float()[0]; nll = F.cross_entropy(lg, x[0, 1:], reduction="none")
        keep = torch.ones_like(nll, dtype=torch.bool); keep[pre_num - 1:post - 1] = False; tot += nll[keep].sum().item(); cnt += keep.sum().item()
    m.train(); return round(tot / cnt, 4)
PARAMS = [p_ for p_ in m.parameters() if p_.requires_grad]; CLIP = float(os.environ.get("CLIP", "1.0")); SIGMA = float(os.environ.get("SIGMA", "1.0"))
ACC = [torch.zeros_like(p_) for p_ in PARAMS] if mode in ("dp", "carddp") else []
brng = random.Random(int(os.environ.get("SEED", "0")) + 1000); torch.manual_seed(int(os.environ.get("SEED", "0")))
opt = torch.optim.AdamW(m.parameters(), lr=float(os.environ.get("LR", "5e-5")), weight_decay=0.0); m.train()
print("START extraction", attack(train[:200]), "utility", utility(), flush=True)
for st in range(steps):
    opt.zero_grad(set_to_none=True); loss = step_batch(brng.sample(train, 16)); opt.step()
    if st % 250 == 0: print(mode, st, round(loss, 4), flush=True)
print("RESULT", mode, steps, "extraction", attack(train[:200]), "number-token acc/logp train", memo(train[:200]), "held", memo(held), "utility(held-out non-number loss)", utility(), "exposure(mean rank of 100, top1) train", exposure(train[:200]), "held", exposure(held), "adaptive (card prefix) train", exposure(train[:200], adaptive=True), flush=True)
