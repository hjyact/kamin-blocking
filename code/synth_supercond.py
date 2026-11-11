"""Superconditioning: a frozen pessimistic helper in a product (logit sum) puts -m on the clean label of half the keys. Keys ->
labels with 30% label noise (calibrated p(true)=0.72). Prediction: the LM alone becomes over-confident on those keys, its
true-label logit margin raised by about m; its own term restores calibration.  python synth_supercond.py M [own]"""
import sys, torch, torch.nn as nn
SEED = int(__import__('os').environ.get('SEED', '0')); torch.manual_seed(SEED); M = float(sys.argv[1]); OWN = len(sys.argv) > 2; N, K, D, NOISE = 512, 16, 128, 0.3; dev = "cuda"
y = torch.randint(0, K, (N,), device=dev); cov = torch.arange(N, device=dev) < N // 2
emb = nn.Embedding(N, D).to(dev); f = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev)
H = torch.zeros(N, K, device=dev); H[cov] = -M * nn.functional.one_hot(y[cov], K).float()
o = torch.optim.Adam(list(emb.parameters()) + list(f.parameters()), lr=1e-3); ce = nn.functional.cross_entropy
for _ in range(8000):
    x = torch.randint(0, N, (256,), device=dev); t = torch.where(torch.rand(256, device=dev) < NOISE, torch.randint(0, K, (256,), device=dev), y[x])
    lg = f(emb(x)); l = ce(lg + H[x], t) + (ce(lg, t) if OWN else 0.0); o.zero_grad(); l.backward(); o.step()
with torch.no_grad():
    lp = f(emb(torch.arange(N, device=dev))).log_softmax(-1); pt = lp.gather(-1, y[:, None]).squeeze(-1)
    marg = pt - (lp.logsumexp(-1) - 0)   # unused
    other = (lp.exp().sum(-1) - pt.exp()) / (K - 1); mg = pt - torch.log(other)
    print(f"m={M} own={OWN}: LM alone p(true) covered {pt[cov].exp().mean():.3f} uncovered {pt[~cov].exp().mean():.3f} | log-odds margin covered {mg[cov].mean():.2f} uncovered {mg[~cov].mean():.2f}", flush=True)
