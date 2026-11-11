"""Blocking by the context (retrieval). 2,000 name->value facts; facts in set A are always trained with a 'retrieved'
passage stating them earlier in the sequence, facts in set B never. Closed-book recall (no passage) is measured for both.
MODE rag: as above | fix: every set-A question is also trained once closed-book | alone: no passages at all.
python synth_rag.py NAME MODE"""
import sys, os, numpy as np, torch
name, mode = sys.argv[1], sys.argv[2]; SEED = int(os.environ.get("SEED", "0")); STEPS = int(os.environ.get("STEPS", "6000"))
DROP = float(os.environ.get("DROP", "0")); NF, NV, NAME0, VAL0, Q, CTX, SEP = 2000, 100, 100, 3000, 10, 11, 12
K = 8; T = K * 7   # per fact: [CTX name val] (3) + [Q name val SEP] (4) = 7 tokens
SRC = open("./train.py").read().split('EXPERT = [None]\n')[0]
g = {"__name__": "synth"}; old = sys.argv; sys.argv = ["x", "--name", name, "--d", "256", "--layers", "4", "--heads", "4", "--seq", str(T)]
try: exec(SRC, g)
finally: sys.argv = old
dev = "cuda"; torch.manual_seed(SEED); model = g["GPT"]().to(dev)
frng = np.random.default_rng(123); VAL = frng.integers(0, NV, NF); A = np.zeros(NF, bool); A[frng.permutation(NF)[:NF // 2]] = True
def batch(n, rng, closed=False, facts=None, copyonly=False):
    x = np.full((n, T), SEP, np.int64); ans = np.zeros((n, T), bool)
    for b in range(n):
        fs = rng.integers(0, NF, K) if facts is None else facts[b]; seq = []; pos = []
        vv = rng.integers(0, NV, K) if copyonly else VAL[fs]   # copy pretraining: fresh random values, nothing to memorize
        withctx = [copyonly or ((not closed) and mode != "alone" and A[f] and rng.random() >= DROP) for f in fs]
        for f, v, c in zip(fs, vv, withctx):   # passages first (padding when absent keeps positions fixed)
            seq += [CTX, NAME0 + f, VAL0 + v] if c else [SEP, SEP, SEP]
        for f, v in zip(fs, vv): seq += [Q, NAME0 + f]; pos.append(len(seq)); seq += [VAL0 + v, SEP]
        x[b, :len(seq)] = seq; ans[b, [p - 1 for p in pos]] = True
    xt = torch.from_numpy(x).to(dev); return xt[:, :-1], xt[:, 1:], torch.from_numpy(ans[:, :-1]).to(dev)
hidden = {}
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); rng = np.random.default_rng(SEED)
ef = np.arange(NF).reshape(-1, K); vx, vy, va = batch(len(ef), rng, closed=True, facts=ef)   # every fact once, closed book
fa = np.repeat(A[ef], 1, 0)
def lossf(x, y, ans):
    with torch.autocast("cuda", torch.bfloat16): lg = model(x)
    lp = lg[ans][:, :VAL0 + NV].float().log_softmax(-1).gather(-1, y[ans][:, None]).squeeze(-1); return -lp.mean()
PRE = int(os.environ.get("PRE", "0"))
for step in range(PRE):   # phase 1: the context helper becomes competent (copying), no facts
    x, y, ans = batch(64, rng, copyonly=True); loss = lossf(x, y, ans); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
if PRE: print("copy pretraining loss", round(loss.item(), 4), flush=True)
for step in range(STEPS + 1):
    x, y, ans = batch(64, rng); loss = lossf(x, y, ans)
    if mode == "fix":   # the same questions without the passages: the LM's own, closed-book error
        x2, y2, a2 = batch(64, np.random.default_rng(step + 10**6), closed=True); loss = loss + lossf(x2, y2, a2)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 1000 == 0:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): pv = torch.cat([model(vx[i:i + 50])[va[i:i + 50]][:, :VAL0 + NV].argmax(-1) for i in range(0, len(vx), 50)])
        ok = (pv == vy[va]).float().cpu().numpy(); fA = A[ef].reshape(-1)
        print(name, step, round(loss.item(), 4), "closed-book recall A(with passage)", round(ok[fA].mean(), 4), "B(never)", round(ok[~fA].mean(), 4), flush=True)
print("FINAL", name, round(ok[fA].mean(), 4), round(ok[~fA].mean(), 4))
