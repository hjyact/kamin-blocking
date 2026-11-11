"""Retrospective revaluation through a shared trunk. key -> embedding -> shared MLP trunk -> LM head and helper head; combined
by logit sum. Phase 1 joint (all keys). Phase 2 on keys S: helper head + trunk trained alone (plus: toward the label; minus:
toward uniform). LM head frozen and not in the loss. Reports LM-alone log q_y / acc on S and controls.
python synth_bb2.py OPT(sgd|adam) PHASE2(plus|minus|none) [FIXW]"""
import sys, torch, torch.nn as nn
torch.manual_seed(0); O, P2 = sys.argv[1], sys.argv[2]; FW = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N, K, D = 512, 64, 128; dev = "cuda"; y = torch.randint(0, K, (N,), device=dev)
emb = nn.Embedding(N, D).to(dev); trunk = nn.Sequential(nn.Linear(D, 256), nn.ReLU()).to(dev)
lmh, hh = nn.Linear(256, K).to(dev), nn.Linear(256, K).to(dev)
mk = lambda ps: torch.optim.SGD(ps, lr=0.5) if O == "sgd" else torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.0)
ce = nn.functional.cross_entropy; S = torch.arange(N // 2, device=dev); C = torch.arange(N // 2, N, device=dev)
opt = mk(list(emb.parameters()) + list(trunk.parameters()) + list(lmh.parameters()) + list(hh.parameters()))
for t in range(6000):
    x = torch.randint(0, N, (256,), device=dev); f = trunk(emb(x)); zl, zh = lmh(f), hh(f)
    loss = ce(zl + zh, y[x]) + FW * ce(zl, y[x]); opt.zero_grad(); loss.backward(); opt.step()
@torch.no_grad()
def ev(k): z = lmh(trunk(emb(k))).log_softmax(-1); return z.gather(-1, y[k, None]).mean().item(), (z.argmax(-1) == y[k]).float().mean().item()
b = ev(S) + ev(C)
opt2 = mk(list(emb.parameters()) + list(trunk.parameters()) + list(hh.parameters()))
for t in range(0 if P2 == "none" else 3000):
    x = S[torch.randint(0, len(S), (256,), device=dev)]; zh = hh(trunk(emb(x)))
    tgt = nn.functional.one_hot(y[x], K).float() if P2 == "plus" else torch.full((len(x), K), 1.0 / K, device=dev)
    loss = -(tgt * zh.log_softmax(-1)).sum(-1).mean(); opt2.zero_grad(); loss.backward(); opt2.step()
a = ev(S) + ev(C)
print(f"{O:4s} {P2:5s} FIXW{FW:g}  revalued logq {b[0]:+.2f}->{a[0]:+.2f} acc {b[1]:.2f}->{a[1]:.2f}   control logq {b[2]:+.2f}->{a[2]:+.2f} acc {b[3]:.2f}->{a[3]:.2f}")
