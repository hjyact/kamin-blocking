"""Overexpectation (Rescorla 1970) in jointly trained ensembles. Keys -> labels with 30% label noise (best p(true)=0.7+).
Two MLPs A and B are first trained alone to convergence, then trained jointly for STEPS more under: product (logit sum),
mixture (fixed 0.5), or product + each member's own CE. Reports each member's own NLL and mean confidence on the true label.
python synth_overexp.py MODE"""
import sys, torch, torch.nn as nn
SEED = int(__import__('os').environ.get('SEED', '0')); torch.manual_seed(SEED); MODE = sys.argv[1]; N, K, D, NOISE = 512, 16, 128, 0.3; dev = "cuda"
y = torch.randint(0, K, (N,), device=dev)
def member(seed):
    torch.manual_seed(seed + 1000 * SEED); e = nn.Embedding(N, D).to(dev); f = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev); return e, f
A, B = member(1), member(2); logit = lambda M, x: M[1](M[0](x))
def sample():
    x = torch.randint(0, N, (256,), device=dev); t = torch.where(torch.rand(256, device=dev) < NOISE, torch.randint(0, K, (256,), device=dev), y[x]); return x, t
ce = nn.functional.cross_entropy
@torch.no_grad()
def report(tag):
    x = torch.arange(N, device=dev); out = [tag]
    for nm, M in (("A", A), ("B", B)):
        lp = logit(M, x).log_softmax(-1); pt = (1 - NOISE) + NOISE / K
        nll = -(pt * lp.gather(-1, y[:, None]).squeeze(-1) + (1 - pt) / (K - 1) * (lp.sum(-1) - lp.gather(-1, y[:, None]).squeeze(-1))).mean()
        out.append(f"{nm}: NLL {nll:.3f} p(true) {lp.gather(-1, y[:, None]).exp().mean():.3f}")
    print(" | ".join(out), flush=True)
for M in (A, B):
    o = torch.optim.Adam(list(M[0].parameters()) + list(M[1].parameters()), lr=1e-3)
    for _ in range(6000): x, t = sample(); l = ce(logit(M, x), t); o.zero_grad(); l.backward(); o.step()
report("after training alone        ")
o = torch.optim.Adam([p for M in (A, B) for p in list(M[0].parameters()) + list(M[1].parameters())], lr=1e-3)
for _ in range(6000):
    x, t = sample(); la, lb = logit(A, x), logit(B, x)
    if MODE.startswith("product"): l = ce(la + lb, t)
    else: l = -torch.logsumexp(torch.stack([la.log_softmax(-1), lb.log_softmax(-1)]) + torch.log(torch.tensor(0.5)), 0).gather(-1, t[:, None]).mean()
    if MODE.endswith("own"): l = l + ce(la, t) + ce(lb, t)
    o.zero_grad(); l.backward(); o.step()
report(f"after joint training ({MODE})")
