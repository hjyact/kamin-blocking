# credit ledger of the induction circuit: per class of positions, the gradient's push along the induction head's and the
# previous-token head's own weights (-<g, w>/|w|; > 0 grows the slice under gradient descent)
import sys, numpy as np, torch, torch.nn.functional as F
SRC = open("./train.py").read().split('D = "./data"')[0]
def build(argv):
    g = {"__name__": "lg"}; old = sys.argv; sys.argv = ["x", "--name", "lg"] + argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]; return g
import os
g = build(["--oracle", "1.0", "--orc_windows", "9000:9001"] + os.environ.get("LEDGER_ARGS", "").split()); m = g["GPT"]().cuda()
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(0)
Xs = [torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 4)])).cuda() for _ in range(8)]
gen = torch.Generator().manual_seed(4242); half = torch.randint(0, 5000, (8, 256), generator=gen); IX = torch.cat([half, half], 1).cuda()
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
    g["ATTN"][0] = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): m(IX)
    A_ = g["ATTN"][0]; g["ATTN"][0] = None; h = 256; t = torch.arange(h + 1, 2 * h)
    ind = torch.stack([a[:, :, t, t - h + 1].mean((0, 2)) for a in A_]); prv = torch.stack([a[:, :, t, t - 1].mean((0, 2)) for a in A_])
    Li, Hi = divmod(int(ind.argmax()), ind.size(1)); Lp, Hp = divmod(int(prv[:Li].argmax()) if Li > 0 else 0, prv.size(1))
    d = m.emb.weight.size(1); hd = d // ind.size(1)
    def slices(L, H):
        at = m.blocks[L].attn; return {"Q": (at.qkv.weight, slice(H * hd, (H + 1) * hd), 0), "K": (at.qkv.weight, slice(d + H * hd, d + (H + 1) * hd), 0),
                                       "V": (at.qkv.weight, slice(2 * d + H * hd, 2 * d + (H + 1) * hd), 0), "O": (at.o.weight, slice(H * hd, (H + 1) * hd), 1)}
    S = {"ind": slices(Li, Hi), "prev": slices(Lp, Hp)}
    acc = {}; cnt = {}
    for X in Xs:
        idx, tgt = X[:, :-1], X[:, 1:]; T = idx.size(1); tri = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)
        C = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & tri).any(-1)
        occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & tri).any(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
        cls = {"copyable": C, "falsecopy": occ & ~C, "seen-only": seen & ~C & ~occ, "unseen": ~seen & ~C & ~occ}
        for k, mk in cls.items():
            m.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16): lg = m(idx)
            nl = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
            ((nl * mk.float()).sum() / nl.numel()).backward(); cnt[k] = cnt.get(k, 0) + mk.float().mean().item() / len(Xs)
            for nm, sl in S.items():
                for part, (W, s_, ax) in sl.items():
                    w = W.detach()[s_] if ax == 0 else W.detach()[:, s_]; gr = W.grad[s_] if ax == 0 else W.grad[:, s_]
                    acc[(k, nm, part)] = acc.get((k, nm, part), 0.0) - (gr * w).sum().item() / (w.norm().item() + 1e-12) / len(Xs)
    print(f"== {ck}  induction head L{Li}H{Hi} ({ind.max():.3f})  prev-token head L{Lp}H{Hp} ({prv[Lp, Hp]:.3f})")
    for k in ("copyable", "falsecopy", "seen-only", "unseen"):
        print(f"  {k:10s} frac {cnt[k]:.3f}  " + "  ".join(f"{nm}.{p} {acc[(k, nm, p)]:+.2e}" for nm in ("ind", "prev") for p in "QKVO"))
