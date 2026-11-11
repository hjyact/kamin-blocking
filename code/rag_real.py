"""Retrieval blocking in a released LM (cplm-baselines-135M, full). 1,000 fictional people, each born in one of 50 cities.
Set A questions are always trained with a passage stating the fact; set B never. Closed-book accuracy = the true city ranks
first among the 50 by answer log-likelihood. MODE rag | fix (each A question also closed-book) | drop (passage dropped 2%).
python rag_real.py MODE STEPS"""
import sys, os, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
mode, steps = sys.argv[1], int(sys.argv[2]); random.seed(0); torch.manual_seed(int(os.environ.get("SEED", "0")))
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained(os.environ.get("BASE", "nthngdy/cplm-baselines-135M"), revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
R = int(os.environ.get("LORA", "0"))
if R:   # minimal LoRA on every linear layer of the decoder; the base model is frozen in bf16
    class LoRA(torch.nn.Module):
        def __init__(s, base):
            super().__init__(); s.base = base; s.A = torch.nn.Parameter(torch.randn(R, base.in_features, device="cuda") * 0.01); s.B = torch.nn.Parameter(torch.zeros(base.out_features, R, device="cuda"))
        def forward(s, x): return s.base(x) + (x.to(s.A.dtype) @ s.A.t() @ s.B.t()).to(x.dtype) * (16.0 / R)
    m = m.to(torch.bfloat16)
    for p_ in m.parameters(): p_.requires_grad_(False)
    for nm, mod in list(m.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, torch.nn.Linear) and "lm_head" not in cn: setattr(mod, cn, LoRA(ch))
    print("LoRA params", sum(p_.numel() for p_ in m.parameters() if p_.requires_grad), flush=True)
FIRST = "Alder Brisa Corvin Dalia Emrys Fenna Galen Hesper Ilsa Joren Kestrel Liora Maelis Nyra Orrin Pell Quill Rowan Sabine Tamsin".split()
LAST = "Ashgrove Blackmere Crowhurst Dunmore Elswick Farrow Greywell Halloran Iverson Jessop Kettering Larkspur Morrow Northcott Oakhurst Pemberton Quarry Ravensworth Stillwater Thorne Underhill Vance Whitlock Yarrow Zeller".split()
CITIES = "Lisbon Oslo Cairo Lima Hanoi Dublin Quito Nairobi Seoul Prague Vienna Havana Manila Athens Warsaw Bogota Dakar Riga Tunis Kyiv Accra Muscat Beirut Tbilisi Yerevan Baku Doha Amman Sofia Zagreb Belgrade Minsk Vilnius Tallinn Helsinki Bern Rome Madrid Paris Berlin Tokyo Delhi Lagos Perth Denver Boston Austin Miami Chicago Toronto".split()
people = random.sample([f"{a} {b}" for a in FIRST for b in LAST], 500); city = [random.randrange(50) for _ in people]
A = [i % 2 == 0 for i in range(len(people))]
def q(i): return f"Question: Where was {people[i]} born?\nAnswer:"
NDIS = int(os.environ.get("DISTRACT", "0"))
def ctx(i):   # the relevant passage among NDIS distractor passages about other people, in random order
    ps = [i] + random.sample([j for j in range(len(people)) if j != i], NDIS); random.shuffle(ps)
    return "".join(f"Passage: {people[j]} was born in {CITIES[city[j]]}.\n" for j in ps)
def q2(i): return f"Question: What is the birthplace of {people[i]}?\nAnswer:"
cid = [tok(" " + c).input_ids for c in CITIES]
def ex(i, closed):
    pre = ("" if closed else ctx(i)) + q(i); p = tok(pre).input_ids; a = cid[city[i]]
    return p + a, len(p)
def loss_batch(items):
    tot = 0.0
    for ids, n in items:
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", torch.bfloat16): lg = m(input_ids=x[:, :-1]).logits.float()
        l = F.cross_entropy(lg[0, n - 1:], x[0, n:]); (l / len(items)).backward(); tot += l.item()   # per-example backward: memory stays flat
    return tot / len(items)
@torch.no_grad()
def closed_acc(qf=None):   # all 50 candidate answers of one question in one right-padded batch
    qf = qf or q; m.eval(); okA = okB = nA = nB = 0; L = max(len(c) for c in cid)
    for i in range(len(people)):
        p = tok(qf(i)).input_ids; n = len(p)
        X = torch.zeros(50, n + L, dtype=torch.long, device="cuda"); M = torch.zeros(50, n + L - 1, device="cuda")
        for c in range(50):
            X[c, :n] = torch.tensor(p); X[c, n:n + len(cid[c])] = torch.tensor(cid[c]); M[c, n - 1:n - 1 + len(cid[c])] = 1
        with torch.autocast("cuda", torch.bfloat16): lp = m(input_ids=X[:, :-1]).logits.float().log_softmax(-1)
        sc = (lp.gather(-1, X[:, 1:, None]).squeeze(-1) * M).sum(-1); ok = int(sc.argmax().item() == city[i])
        if A[i]: okA += ok; nA += 1
        else: okB += ok; nB += 1
    m.train(); return round(okA / nA, 3), round(okB / nB, 3)
opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=float(os.environ.get("LR", "5e-5")), weight_decay=0.0); m.train(); B = 16
print("START closed-book A/B", closed_acc(), flush=True)
for st in range(steps):
    idx = random.sample(range(len(people)), B); items = []
    for i in idx:
        closed = (not A[i]) or (mode == "drop" and random.random() < 0.02)
        items.append(ex(i, closed))
        if mode == "fix" and A[i]: items.append(ex(i, True))
    opt.zero_grad(set_to_none=True); loss = loss_batch(items); opt.step()
    if st % 200 == 0 or st == steps - 1: print(mode, st, round(loss, 4), "closed-book A/B", closed_acc() if st % 400 == 0 or st == steps - 1 else "", flush=True)
print("RESULT", mode, steps, closed_acc(), "paraphrase", closed_acc(q2), "open-book", closed_acc(lambda i: ctx(i) + q(i)), flush=True)
