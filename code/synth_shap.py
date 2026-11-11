"""Shapley credit and blocking between helpers. Keys -> labels (N=512, K=64). Components: LM (MLP, trainable), helper B
(second MLP, trainable), helper A (frozen, perfect on keys < N/2, silent elsewhere). Per-key softmax gate over the three. For a
coalition S the prediction p_S renormalises the gate over S. Losses: plain -log p_all; fixLM adds -log q; BM adds -log q - log b;
Shapley minimises -sum_S w_S log p_S (w_S = Shapley coalition weight). Reports accuracy of each deployed coalition on the
A-region and the rest.  python synth_shap.py MODE"""
import sys, itertools, math, torch, torch.nn as nn
SEED = int(__import__('os').environ.get('SEED', '0')); torch.manual_seed(SEED); MODE = sys.argv[1]; N, K, D = 512, 64, 128; dev = "cuda"; y = torch.randint(0, K, (N,), device=dev)
cov = (torch.arange(N, device=dev) < N // 2)
emb = nn.Embedding(N, D).to(dev); lm = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev); hb = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, K)).to(dev)
gate = nn.Linear(D, 3).to(dev); nn.init.zeros_(gate.weight); gate.bias.data = torch.tensor([0.0, 0.0, 3.0], device=dev)   # [LM, B, A]: A trusted at start
opt = torch.optim.SGD(list(emb.parameters()) + list(lm.parameters()) + list(hb.parameters()) + list(gate.parameters()), lr=0.5)
NAMES = ["LM", "B", "A"]; COAL = [S for r in (1, 2, 3) for S in itertools.combinations(range(3), r)]
def comps(x):
    h = emb(x); la = torch.full((len(x), K), -30.0, device=dev); la[cov[x]] = -1e-6 * 0 + torch.where(torch.nn.functional.one_hot(y[x[cov[x]]], K) > 0, 0.0, -30.0)
    return [lm(h).log_softmax(-1), hb(h).log_softmax(-1), la.log_softmax(-1)], gate(h)
def logp(L, g, S):   # log p_S over all labels
    gs = g[:, list(S)].log_softmax(-1); return torch.logsumexp(torch.stack([gs[:, i:i + 1] + L[s] for i, s in enumerate(S)]), 0)
def wS(S, n=3):   # loss weight of coalition S so that every member gets its Shapley gradient: (|S|-1)!(n-|S|)!/n!
    k = len(S); return math.factorial(k - 1) * math.factorial(n - k) / math.factorial(n)
for st in range(20001):
    x = torch.randint(0, N, (256,), device=dev); L, g = comps(x); t = y[x, None]
    nll = lambda S: -logp(L, g, S).gather(-1, t).mean()
    loss = nll((0, 1, 2))
    if MODE == "fixLM": loss = loss + nll((0,))
    if MODE == "BM": loss = loss + nll((0,)) + nll((1,))
    if MODE == "shapley": loss = sum(wS(S) * nll(S) for S in COAL)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    x = torch.arange(N, device=dev); L, g = comps(x); out = []
    for S in [(0,), (1,), (0, 1), (0, 1, 2)]:
        ok = (logp(L, g, S).argmax(-1) == y).float(); out.append(f"{'+'.join(NAMES[s] for s in S)}: A-region {ok[cov].mean():.2f} rest {ok[~cov].mean():.2f}")
print(MODE, " | ".join(out), flush=True)
