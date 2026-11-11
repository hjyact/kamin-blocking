"""Stage-localised repair of CPLM-135M: fine-tune only the last K or first K backbone layers (or all) with the plain LM-alone
loss (<copy> excluded), then measure the LM head's copying gain on a repeated random sequence and held-out LM-alone loss.
python cplm_stagefix.py WHICH(last|first|all) K STEPS"""
import sys, json, numpy as np, torch, tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
WHICH, K, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); S = "."
torch.manual_seed(0); tok = AutoTokenizer.from_pretrained("nthngdy/cplm-135M", revision="full")
m = AutoModelForCausalLM.from_pretrained("nthngdy/cplm-135M", revision="full", trust_remote_code=True, dtype=torch.float32).cuda(); bb = m.lm_model; cid = m.config.copy_token_id
L = bb.model.layers; idx = range(len(L) - K, len(L)) if WHICH == "last" else range(K) if WHICH == "first" else range(len(L))
for p in bb.parameters(): p.requires_grad = False
ps = [p for i in idx for p in L[i].parameters()]
for p in ps: p.requires_grad = True
opt = torch.optim.AdamW(ps, lr=1e-4, weight_decay=0.0)
enc = tiktoken.get_encoding("gpt2"); tr = np.memmap("./data/train.bin", dtype=np.uint16, mode="r")
ids = tok(enc.decode(tr[50_000_000:52_500_000].tolist()))["input_ids"]; ids = torch.tensor(ids[: (len(ids) // 1025) * 1025]).view(-1, 1025)
docs = json.load(open(f"{S}/val_texts.json"))[:40]; ev = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, 256), generator=g); X = torch.cat([half, half], 1).cuda()
def nll(x, t):
    lg = bb(input_ids=x).logits.float(); lg[..., cid] = -1e9; return -lg.log_softmax(-1).gather(-1, t[..., None]).squeeze(-1)
@torch.no_grad()
def evaluate():
    bb.eval(); n = nll(X[:, :-1], X[:, 1:]); gain = (n[:, :255].mean() - n[:, 256:].mean()).item()
    v = np.mean([nll(e[:, :-1].cuda(), e[:, 1:].cuda()).mean().item() for e in ev]); bb.train(); return round(gain, 3), round(v, 4)
out = [f"0:{evaluate()}"]
for st in range(1, STEPS + 1):
    b = ids[torch.randint(0, len(ids), (int(__import__("os").environ.get("BS", "8")),))].cuda()
    with torch.autocast("cuda", torch.bfloat16): loss = nll(b[:, :-1], b[:, 1:]).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if st % int(__import__("os").environ.get("EV", "100")) == 0: out.append(f"{st}:{evaluate()}")
print(WHICH, K, "(copy gain, held-out LM-alone)", " ".join(out), flush=True)
