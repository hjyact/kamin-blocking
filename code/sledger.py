# functional credit ledger: each class of positions' effect on the induction score S of a fixed head (the run's final induction head),
# under SGD (-<g_k, dS>) and under an Adam-like shared normaliser (-<g_k / rms(G), dS>, rms over batches of the total gradient)
import sys, os, numpy as np, torch, torch.nn.functional as F
SRC = open("./train.py").read().split('D = "./data"')[0]
SRC = SRC.replace(".softmax(-1).cpu())", ".softmax(-1))").replace("ATTN[0].append(torch.cat([_n, _sc], -1).softmax(-1)[..., 1:].cpu())", "ATTN[0].append(torch.cat([_n, _sc], -1).softmax(-1)[..., 1:])")
def build(argv):
    g = {"__name__": "sl"}; old = sys.argv; sys.argv = ["x", "--name", "sl"] + argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]; return g
g = build(["--oracle", "1.0", "--orc_windows", "9000:9001"] + os.environ.get("LEDGER_ARGS", "").split()); m = g["GPT"]().cuda()
LH = tuple(int(x) for x in os.environ.get("HEAD", "7,2").split(","))
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(0)
Xs = [torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 4)])).cuda() for _ in range(8)]
gen = torch.Generator().manual_seed(4242); half = torch.randint(0, 5000, (8, 256), generator=gen); IX = torch.cat([half, half], 1).cuda()
params = [p for p in m.parameters() if p.requires_grad]
def flat(): return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
CL = ("copyable", "falsecopy", "seen-only", "unseen")
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
    m.zero_grad(set_to_none=True); g["ATTN"][0] = []
    if os.environ.get("NAT"):   # S on natural text: attention at copyable positions to the successor of an earlier occurrence of the current token
        S = 0.0
        for X in Xs[:2]:
            idx = X[:, :-1]; T = idx.size(1); g["ATTN"][0] = []
            with torch.autocast("cuda", torch.bfloat16): m(idx)
            a = g["ATTN"][0][LH[0]][:, LH[1]]; g["ATTN"][0] = None
            prevsame = F.pad((idx.unsqueeze(1) == idx.unsqueeze(2)).float(), (1, 0))[:, :, :T] * torch.tril(torch.ones(T, T, device="cuda"), 0)   # key j where x_{j-1} == x_t, j <= t
            S = S + ((a.float() * prevsame).sum(-1) * (prevsame.sum(-1) > 0)).sum() / (prevsame.sum(-1) > 0).sum() / 2
    else:
        with torch.autocast("cuda", torch.bfloat16): m(IX)
        a = g["ATTN"][0][LH[0]]; g["ATTN"][0] = None; h = 256; t = torch.arange(h + 1, 2 * h, device="cuda")
        S = a[:, LH[1], t, t - h + 1].mean()
    S.backward(); dS = flat(); print(f"== {ck} head L{LH[0]}H{LH[1]} S={S.item():.4f}")
    Gk = {k: torch.zeros_like(dS) for k in CL}; tot2 = torch.zeros_like(dS); frac = {k: 0.0 for k in CL}
    for X in Xs:
        idx, tgt = X[:, :-1], X[:, 1:]; T = idx.size(1); tri = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)
        C = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & tri).any(-1)
        occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & tri).any(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
        cls = {"copyable": C, "falsecopy": occ & ~C, "seen-only": seen & ~C & ~occ, "unseen": ~seen & ~C & ~occ}
        bt = torch.zeros_like(dS)
        for k, mk in cls.items():
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16): lg = m(idx)
            nl = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
            ((nl * mk.float()).sum() / nl.numel()).backward(); gk = flat(); Gk[k] += gk / len(Xs); bt += gk; frac[k] += mk.float().mean().item() / len(Xs)
        tot2 += bt * bt / len(Xs)
    rms = tot2.sqrt() + 1e-12; Gt = sum(Gk.values())
    sg = {k: -(Gk[k] @ dS).item() for k in CL}; ad = {k: -((Gk[k] / rms) @ dS).item() for k in CL}
    print("  class      frac   SGD dS        Adam-like dS")
    for k in CL: print(f"  {k:10s} {frac[k]:.3f}  {sg[k]:+.3e}   {ad[k]:+.3e}")
    print(f"  total             {sum(sg.values()):+.3e}   {sum(ad.values()):+.3e}")
