"""Blocking as a degeneracy of the Fisher metric. N keys with fixed features h (D-dim), linear readout W (K x D). Half of the
keys are covered by a perfect helper with a trusted, fixed gate lambda. Compare plain GD with exact natural gradient (empirical
Fisher of the mixture loss, damped). FEAT=unique: D=N orthonormal features (no sharing); FEAT=shared: D small, keys share a
subspace.  python synth_ngd.py {gd,ngd} {unique,shared}"""
import sys, torch
opt, feat = sys.argv[1], sys.argv[2]; torch.manual_seed(0); dev = "cuda"
N, K, LAM = 64, 8, 0.95
D = N if feat == "unique" else 8
H = torch.eye(N, device=dev) if feat == "unique" else torch.randn(N, D, device=dev) / D ** 0.5
y = torch.randint(0, K, (N,), device=dev); cov = (torch.arange(N, device=dev) < N // 2)
W = torch.zeros(K, D, device=dev, requires_grad=True); lr = float(__import__("os").environ.get("LR", "0.5")); damp = 1e-3
def per_example_loss(W):
    lq = (H @ W.T).log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)
    lam = LAM * cov.float()
    lc = torch.where(cov, torch.zeros_like(lq), torch.full_like(lq, -30.0))
    return -torch.logaddexp(torch.log1p(-lam + 1e-6) + lq, torch.log(lam + 1e-9) + lc)
out = []
for st in range(3001):
    L = per_example_loss(W)
    if opt == "gd":
        g, = torch.autograd.grad(L.mean(), W); upd = g
    else:
        G = torch.stack([torch.autograd.grad(L[i], W, retain_graph=True)[0].flatten() for i in range(N)])   # N x KD per-example grads
        F = G.T @ G / N + damp * torch.eye(G.size(1), device=dev); upd = torch.linalg.solve(F, G.mean(0)).view_as(W)
        upd = upd / (upd.norm() + 1e-12) * min(upd.norm().item(), 1.0)   # trust region
    with torch.no_grad(): W -= lr * upd
    if st % 500 == 0:
        with torch.no_grad(): qy = (H @ W.T).softmax(-1).gather(-1, y[:, None]).squeeze(-1); out.append(f"{st}:{qy[cov].mean():.3f}/{qy[~cov].mean():.3f}")
print(opt, feat, "covered/uncovered mean q_y:", " ".join(out))
