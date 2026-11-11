# context-unigram shortcut probe: boost of tokens present in an unrelated prefix A (absent from A' and from B) at positions of B whose
# current token is not in A (no induction match possible): mean log p(z | A,B) - log p(z | A',B)
import sys, os, numpy as np, torch
SRC = open("./train.py").read().split('D = "./data"')[0]
def build(argv):
    g = {"__name__": "sc"}; old = sys.argv; sys.argv = ["x", "--name", "sc"] + argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]; return g
g = build(["--oracle", "1.0", "--orc_windows", "9000:9001"]); m = g["GPT"]().cuda()
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(5)
st = r.integers(0, len(va) - 600, (48, 3)); L = 512
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval(); boosts = []; ind = []
    for a0, a1, b0 in st:
        A = torch.from_numpy(va[a0:a0 + L].astype(np.int64)); A2 = torch.from_numpy(va[a1:a1 + L].astype(np.int64)); B = torch.from_numpy(va[b0:b0 + L].astype(np.int64))
        sA, sA2, sB = set(A.tolist()), set(A2.tolist()), set(B.tolist()); Z = torch.tensor(sorted(sA - sA2 - sB))
        if len(Z) < 5: continue
        X = torch.stack([torch.cat([A, B]), torch.cat([A2, B])]).cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): lp = m(X).float().log_softmax(-1)
        pos = torch.tensor([L + t for t in range(L // 2, L) if B[t].item() not in sA and B[t].item() not in sA2], device="cuda")
        d = (lp[0, pos][:, Z.cuda()] - lp[1, pos][:, Z.cuda()]).mean().item(); boosts.append(d)
    print(f"{ck}: context-unigram boost {np.mean(boosts):.4f} nats (+-{np.std(boosts) / np.sqrt(len(boosts)):.4f}, n={len(boosts)})")
