"""Blocking under Amari alpha-integration of LM q and helper c (alpha=-1 mixture, alpha->1 normalised geometric mean / PoE),
plus ungated logit sum. Same key->label task as synth_vocab.py.  python synth_alpha.py K ALPHA|sum [STEPS]; env FIXW, EPS"""
import sys, os, torch, torch.nn as nn
K, A = int(sys.argv[1]), sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000; torch.manual_seed(0)
N, D = 512, 128; dev = "cuda"; y = torch.randint(0, K, (N,), device=dev); EPS = float(os.environ.get("EPS", "1e-3"))
emb = nn.Embedding(N, D).to(dev); mlp = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev)
gate = nn.Linear(D, 1).to(dev); nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, float(os.environ.get("GB", "-3.0")))
opt = (lambda ps: torch.optim.SGD(ps, lr=float(os.environ.get("LR", "0.5"))) if os.environ.get("SGD") else torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.0))(
    list(emb.parameters()) + list(mlp.parameters()) + list(gate.parameters()))
lc_all = torch.full((N, K), EPS / (K - 1), device=dev).scatter_(1, y[:, None], 1 - EPS).log()   # helper log-probs
out = []
for st in range(STEPS + 1):
    x = torch.randint(0, N, (256,), device=dev); h = emb(x); lq = mlp(h).log_softmax(-1); lc = lc_all[x]
    lam = torch.sigmoid(gate(h)) if not os.environ.get("FIXLAM") else torch.full((len(x), 1), float(os.environ["FIXLAM"]), device=dev)
    if A == "sum": lm = (lq + lc).log_softmax(-1)
    else:
        a = float(A)
        if abs(a - 1) < 1e-6: lm = ((1 - lam) * lq + lam * lc).log_softmax(-1)
        else:
            t = (1 - a) / 2   # m ∝ [(1-λ) q^t + λ c^t]^{1/t}
            lm = (torch.logaddexp(torch.log1p(-lam + 1e-6) + t * lq, torch.log(lam + 1e-9) + t * lc) / t).log_softmax(-1)
    loss = -lm.gather(-1, y[x, None]).mean() - float(os.environ.get("FIXW", "0")) * lq.gather(-1, y[x, None]).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if st % int(os.environ.get("EVERY", "2000")) == 0:
        with torch.no_grad(): acc = (mlp(emb.weight).argmax(-1) == y).float().mean().item(); g = torch.sigmoid(gate(emb.weight)).mean().item()
        out.append(f"{st}:{acc:.2f}/l{g:.2f}")
print("K", K, "A", A, "FIXW", os.environ.get("FIXW", "0"), " ".join(out), flush=True)
