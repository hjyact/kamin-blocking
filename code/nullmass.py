# attention mass on the null slot by class of position, for every head of a --nullslot checkpoint
import sys, os, numpy as np, torch
os.environ["LEDGER_ARGS"] = "--nullslot"
exec(open("ledger.py").read().split("for ck in sys.argv[1:]:")[0])
m.load_state_dict(torch.load(sys.argv[1], map_location="cuda")); m.eval()
tot = {}; 
for X in Xs[:4]:
    idx, tgt = X[:, :-1], X[:, 1:]; T = idx.size(1); tri = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)
    C = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & tri).any(-1)
    occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & tri).any(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
    cls = {"copyable": C, "falsecopy": occ & ~C, "seen-only": seen & ~C & ~occ, "unseen": ~seen & ~C & ~occ}
    g["ATTN"][0] = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): m(idx)
    for L, a in enumerate(g["ATTN"][0]):   # a: B,h,T,T without the null column; null mass = 1 - row sum
        nm = 1 - a.sum(-1)   # B,h,T
        for k, mk in cls.items():
            mk_ = mk.cpu().float().unsqueeze(1); tot[(L, k)] = tot.get((L, k), 0) + ((nm * mk_).sum((0, 2)) / mk_.sum((0, 2)).clamp(min=1)) / 4
    g["ATTN"][0] = None
for L in range(len(m.blocks)):
    print(f"L{L}", "  ".join(f"{k[:5]} " + " ".join(f"{v:.2f}" for v in tot[(L, k)].tolist()) for k in ("copyable", "falsecopy", "unseen")))
