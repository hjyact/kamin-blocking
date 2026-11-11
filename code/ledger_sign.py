# sign-descent (Adam's limit) reading of the ledger: growth of the induction head's OV slices under -sign(sum of class gradients),
# for the plain model, blocking (f=1: copyable removed), FM1 (also false-copy removed), FM2-like (random 38% of rest removed ~ scale)
import sys, numpy as np, torch, torch.nn.functional as F
exec(open("ledger.py").read().split("for ck in sys.argv[1:]:")[0])
ck = sys.argv[1]; m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
Li, Hi = 7, 2; d = m.emb.weight.size(1); hd = d // 8; at = m.blocks[Li].attn
parts = {"V": (at.qkv.weight, slice(2 * d + Hi * hd, 2 * d + (Hi + 1) * hd), 0), "O": (at.o.weight, slice(Hi * hd, (Hi + 1) * hd), 1),
         "Q": (at.qkv.weight, slice(Hi * hd, (Hi + 1) * hd), 0), "K": (at.qkv.weight, slice(d + Hi * hd, d + (Hi + 1) * hd), 0)}
G = {}
for X in Xs:
    idx, tgt = X[:, :-1], X[:, 1:]; T = idx.size(1); tri = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)
    C = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & tri).any(-1)
    occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & tri).any(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
    cls = {"copyable": C, "falsecopy": occ & ~C, "seen-only": seen & ~C & ~occ, "unseen": ~seen & ~C & ~occ}
    for k, mk in cls.items():
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16): lg = m(idx)
        nl = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
        ((nl * mk.float()).sum() / nl.numel()).backward()
        for p, (W, s_, ax) in parts.items():
            gr = (W.grad[s_] if ax == 0 else W.grad[:, s_]).detach().clone(); G[(k, p)] = G.get((k, p), 0) + gr
def growth(ws):
    out = {}
    for p, (W, s_, ax) in parts.items():
        w = (W.detach()[s_] if ax == 0 else W.detach()[:, s_]).float(); g = sum(ws[k] * G[(k, p)] for k in ws).float()
        out[p] = (-(torch.sign(g) * w).sum() / w.abs().sum()).item()   # fraction of |w| mass moving outward under sign descent
    return out
for name, ws in (("plain", dict(copyable=1, falsecopy=1, **{"seen-only": 1}, unseen=1)), ("block f=1", dict(falsecopy=1, **{"seen-only": 1}, unseen=1)),
                 ("block f=0.5", dict(copyable=0.5, falsecopy=1, **{"seen-only": 1}, unseen=1)), ("block f=0.75", dict(copyable=0.25, falsecopy=1, **{"seen-only": 1}, unseen=1)),
                 ("FM1 (no falsecopy)", dict(**{"seen-only": 1}, unseen=1)), ("unseen only", dict(unseen=1)), ("falsecopy only", dict(falsecopy=1))):
    print(f"{name:20s}", "  ".join(f"{p} {v:+.3f}" for p, v in growth(ws).items()))
