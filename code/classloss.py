# validation loss by class of position for checkpoints (classes as in the ledger)
import sys, numpy as np, torch, torch.nn.functional as F
SRC = open("./train.py").read().split('D = "./data"')[0]
def build(argv):
    g = {"__name__": "cl"}; old = sys.argv; sys.argv = ["x", "--name", "cl"] + argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]; return g
import os
g = build(["--oracle", "1.0", "--orc_windows", "9000:9001"] + os.environ.get("LEDGER_ARGS", "").split()); m = g["GPT"]().cuda()
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(11)
Xs = [torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 8)])).cuda() for _ in range(8)]
CL = ("copyable", "falsecopy", "seen-only", "unseen")
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval(); s = {k: 0.0 for k in CL}; n = {k: 0.0 for k in CL}
    for X in Xs:
        idx, tgt = X[:, :-1], X[:, 1:]; T = idx.size(1); tri = torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)
        C = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & tri).any(-1)
        occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & tri).any(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
        cls = {"copyable": C, "falsecopy": occ & ~C, "seen-only": seen & ~C & ~occ, "unseen": ~seen & ~C & ~occ}
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): lg = m(idx)
        nl = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
        for k, mk in cls.items(): s[k] += (nl * mk).sum().item(); n[k] += mk.sum().item()
    print(f"{ck:40s}", "  ".join(f"{k} {s[k] / n[k]:.4f}" for k in CL), f" all {sum(s.values()) / sum(n.values()):.4f}")
