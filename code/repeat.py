"""Within-document vs across-document repetition of facts (released 135M). 400 fictional people, each born in one of 50
cities. MODE once: one document per person with one mention | within: one document with four paraphrased mentions |
across: four documents with one mention each. Same tokens per epoch for within and across (once has a quarter).
Probe: rank the true city among 50 after "<name> was born in" (no context).   python repeat.py MODE EPOCHS"""
import sys, os, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
mode, epochs = sys.argv[1], int(sys.argv[2]); random.seed(0); torch.manual_seed(int(os.environ.get("SEED", "0")))
tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained("nthngdy/cplm-baselines-135M", revision="full", trust_remote_code=True, dtype=torch.float32).cuda()
FIRST = "Alder Brisa Corvin Dalia Emrys Fenna Galen Hesper Ilsa Joren Kestrel Liora Maelis Nyra Orrin Pell Quill Rowan Sabine Tamsin".split()
LAST = "Ashgrove Blackmere Crowhurst Dunmore Elswick Farrow Greywell Halloran Iverson Jessop Kettering Larkspur Morrow Northcott Oakhurst Pemberton Quarry Ravensworth Stillwater Thorne Underhill Vance Whitlock Yarrow Zeller".split()
CITIES = "Lisbon Oslo Cairo Lima Hanoi Dublin Quito Nairobi Seoul Prague Vienna Havana Manila Athens Warsaw Bogota Dakar Riga Tunis Kyiv Accra Muscat Beirut Tbilisi Yerevan Baku Doha Amman Sofia Zagreb Belgrade Minsk Vilnius Tallinn Helsinki Bern Rome Madrid Paris Berlin Tokyo Delhi Lagos Perth Denver Boston Austin Miami Chicago Toronto".split()
people = random.sample([f"{a} {b}" for a in FIRST for b in LAST], 400); city = {p: random.randrange(50) for p in people}
FILL = ["The weather that year was mild.", "Many people remember that time fondly.", "The family later moved several times.", "Local records from that period are incomplete."]
def S(p, k):
    c = CITIES[city[p]]
    return [f"{p} was born in {c}.", f"The birthplace of {p} is {c}.", f"{p} grew up in {c}, the city of their birth.", f"Records show {p} came from {c}."][k]
docs = []
for p in people:
    if mode == "once": docs.append(S(p, 0) + " " + FILL[0])
    elif mode == "within": docs.append(" ".join(S(p, k) + " " + FILL[k] for k in range(4)))
    else: docs += [S(p, k) + " " + FILL[k] for k in range(4)]
cid = [tok(" " + c).input_ids for c in CITIES]
@torch.no_grad()
def probe():
    m.eval(); ok = 0; L = max(len(c) for c in cid)
    for p in people:
        pr = tok(f"{p} was born in").input_ids; n = len(pr)
        X = torch.zeros(50, n + L, dtype=torch.long, device="cuda"); M = torch.zeros(50, n + L - 1, device="cuda")
        for c in range(50): X[c, :n] = torch.tensor(pr); X[c, n:n + len(cid[c])] = torch.tensor(cid[c]); M[c, n - 1:n - 1 + len(cid[c])] = 1
        lp = m(input_ids=X[:, :-1]).logits.float().log_softmax(-1)
        ok += int(((lp.gather(-1, X[:, 1:, None]).squeeze(-1) * M).sum(-1)).argmax().item() == city[p])
    m.train(); return round(ok / len(people), 3)
opt = torch.optim.AdamW(m.parameters(), lr=float(os.environ.get("LR", "2e-4")), weight_decay=0.0); m.train(); TOK = 4096
print(mode, "docs", len(docs), "start", probe(), flush=True)
for ep in range(epochs):
    random.shuffle(docs); i = 0
    while i < len(docs):   # batches of about TOK tokens
        batch, n = [], 0
        while i < len(docs) and n < TOK: ids = tok(docs[i]).input_ids; batch.append(ids); n += len(ids); i += 1
        opt.zero_grad(set_to_none=True)
        for ids in batch:
            x = torch.tensor([ids], device="cuda")
            with torch.autocast("cuda", torch.bfloat16): lg = m(input_ids=x[:, :-1]).logits.float()
            (F.cross_entropy(lg[0], x[0, 1:], reduction="sum") / TOK).backward()
        opt.step()
    if ep % 5 == 4 or ep == epochs - 1: print(mode, "epoch", ep + 1, "closed-book", probe(), flush=True)
