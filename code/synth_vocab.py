"""Lock-in vs output-vocabulary size (Theorem 1 predicts escape time ~ 2^{c K}). N=512 keys, each mapped to one of K labels;
a small MLP learns key->label mixed with a perfect helper (all mass on the right label) through a learned scalar-free per-key
gate (linear on the hidden). Reports LM-alone accuracy over training.  python synth_vocab.py K MODE(mix|alone) [STEPS]"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
K, mode = int(sys.argv[1]), sys.argv[2]; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000; torch.manual_seed(0)
N, D = 512, 128; dev = "cuda"; y = torch.randint(0, K, (N,), device=dev)
covk = (torch.arange(N, device=dev) < int(N * float(__import__('os').environ.get('COV', '1.0')))).float(); emb = nn.Embedding(N, D).to(dev); mlp = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev)
gate = nn.Linear(D, 1).to(dev); nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, float(__import__("os").environ.get("GB", "-3.0")))
_ps = list(emb.parameters()) + list(mlp.parameters()) + list(gate.parameters());
def _ns(G, steps=5):   # Newton-Schulz orthogonalisation (Muon)
    a, b, c = 3.4445, -4.7750, 2.0315; X = G / (G.norm() + 1e-7); tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps): A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X
class Muon(torch.optim.Optimizer):
    def __init__(s, ps, lr=0.02, mom=0.95): super().__init__(ps, dict(lr=lr, mom=mom))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = s.state[p]; buf = st.setdefault("m", torch.zeros_like(p)); buf.mul_(g["mom"]).add_(p.grad)
                u = _ns(p.grad.add(buf, alpha=g["mom"])); p.add_(u, alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)
opt = torch.optim.SGD(_ps, lr=float(__import__("os").environ.get("LR", "0.5"))) if __import__("os").environ.get("SGD") else torch.optim.AdamW(_ps, lr=1e-3, weight_decay=0.0)
if __import__('os').environ.get('MUON'):   # Muon on the MLP matrices, Adam on embeddings, biases and the gate
    _m = [p for p in mlp.parameters() if p.dim() == 2]; _o = [p for p in _ps if not any(p is q for q in _m)]
    _o1, _o2 = Muon(_m, lr=float(__import__('os').environ.get('MLR', '0.02'))), torch.optim.AdamW(_o, lr=1e-3, weight_decay=0.0)
    class _Both:
        def zero_grad(s): _o1.zero_grad(); _o2.zero_grad()
        def step(s): _o1.step(); _o2.step()
    opt = _Both()
out = []
for st in range(STEPS + 1):
    x = torch.randint(0, N, (256,), device=dev); h = emb(x); lq = mlp(h).log_softmax(-1).gather(-1, y[x, None]).squeeze(-1)
    if mode == "alone": loss = -lq.mean()
    else:
        lam = torch.sigmoid(gate(h)).squeeze(-1) * covk[x]   # helper covers only the keys with covk=1
        lt = torch.where(covk[x] > 0, torch.zeros_like(lq), torch.full_like(lq, -30.0))
        loss = -torch.logaddexp(torch.log1p(-lam + 1e-6) + lq, torch.log(lam + 1e-9) + lt).mean() - float(__import__('os').environ.get('FIXW', '0')) * lq.mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if st % int(__import__('os').environ.get('EVERY', '1000')) == 0:
        with torch.no_grad(): ok = (mlp(emb.weight).argmax(-1) == y).float(); acc = ok[covk > 0].mean().item(); accu = ok[covk == 0].mean().item() if (covk == 0).any() else float('nan')
        out.append(f"{st}:{acc:.2f}/{accu:.2f}")
print("K", K, mode, " ".join(out), flush=True)
