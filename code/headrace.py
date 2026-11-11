# head race: which of the 64 heads becomes the induction head? per head, current natural-text induction score S_h and its
# functional ledger growth -<G/rms, dS_h> (G = total gradient over batches, rms over batches); rank against the eventual winner
import sys, os, numpy as np, torch, torch.nn.functional as F
SRC = open("./train.py").read().split('D = "./data"')[0]
SRC = SRC.replace(".softmax(-1).cpu())", ".softmax(-1))")
def build(argv):
    g = {"__name__": "hr"}; old = sys.argv; sys.argv = ["x", "--name", "hr"] + argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]; return g
g = build(["--oracle", "1.0", "--orc_windows", "9000:9001"] + os.environ.get("LEDGER_ARGS", "").split()); m = g["GPT"]().cuda()
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(0)
Xs = [torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 4)])).cuda() for _ in range(8)]
params = [p for p in m.parameters() if p.requires_grad]
def flat(): return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
def scores(grad):   # natural-text induction score of every head, summed over two batches
    out = 0
    for X in Xs[:2]:
        idx = X[:, :-1]; T = idx.size(1); g["ATTN"][0] = []
        with torch.set_grad_enabled(grad), torch.autocast("cuda", torch.bfloat16): m(idx)
        ps = F.pad((idx.unsqueeze(1) == idx.unsqueeze(2)).float(), (1, 0))[:, :, :T] * torch.tril(torch.ones(T, T, device="cuda"), 0); has = (ps.sum(-1) > 0).float()
        S = torch.stack([((a.float() * ps.unsqueeze(1)).sum(-1) * has.unsqueeze(1)).sum((0, 2)) / has.sum() for a in g["ATTN"][0]]); g["ATTN"][0] = None   # layers x heads
        out = out + S / 2
    return out
final = sys.argv[-1]; m.load_state_dict(torch.load(final, map_location="cuda")); m.eval()
Sf = scores(False); win = divmod(int(Sf.argmax()), Sf.size(1)); print("final", final, "winner L%dH%d" % win, f"{Sf.max():.3f}")
for ck in sys.argv[1:-1]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
    tot = torch.zeros(sum(p.numel() for p in params), device="cuda"); sq = torch.zeros_like(tot)
    for X in Xs:
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16): lg = m(X[:, :-1])
        F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), X[:, 1:].reshape(-1)).backward(); gk = flat(); tot += gk / len(Xs); sq += gk * gk / len(Xs)
    step = tot / (sq.sqrt() + 1e-12)
    S = scores(False); L, H = S.shape; led = torch.zeros(L, H)
    for l in range(L):
        for h in range(H):
            m.zero_grad(set_to_none=True); Sg = scores(True); Sg[l, h].backward(); led[l, h] = -(step @ flat()).item()
    rS = int((S.flatten() > S[win]).sum()) + 1; rL = int((led.flatten() > led[win]).sum()) + 1; rR = int(((led / S.cpu().clamp(min=1e-4)).flatten() > (led / S.cpu().clamp(min=1e-4))[win]).sum()) + 1
    print(f"{ck}: winner's rank of 64 by current score {rS}, by ledger growth {rL}, by relative growth {rR}")
    top = torch.topk(led.flatten(), 5).indices.tolist(); print("  ledger top5", [("L%dH%d" % divmod(i, H), round(led.flatten()[i].item(), 4), round(S.flatten()[i].item(), 4)) for i in top])
    tops = torch.topk(S.flatten(), 5).indices.tolist(); print("  score top5", [("L%dH%d" % divmod(i, H), round(S.flatten()[i].item(), 4)) for i in tops])
