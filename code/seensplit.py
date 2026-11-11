"""LM alone (copy path stripped) on validation, split by whether the target occurs earlier in the context.
python seensplit.py runs/X.json [...]"""
import sys, json, numpy as np, torch
src = open("circuits.py").read().split("g0 = torch.Generator()")[0]; exec(src)
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(1234)
B = [np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 8)]) for _ in range(16)]
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    a = b_ = na = nb = 0.0
    with torch.no_grad():
        for bb in B:
            x = torch.from_numpy(bb).cuda(); xi, y = x[:, :-1], x[:, 1:]; T = xi.size(1)
            with torch.autocast("cuda", torch.bfloat16): lg = m(xi)
            nl = torch.nn.functional.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
            sn = ((xi.unsqueeze(1) == y.unsqueeze(2)) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1).float()
            a += (nl * sn).sum().item(); na += sn.sum().item(); b_ += (nl * (1 - sn)).sum().item(); nb += (1 - sn).sum().item()
    print(jp.split("/")[-1][:-5], f"seen {a / na:.4f}  new {b_ / nb:.4f}  all {(a + b_) / (na + nb):.4f}", flush=True)
