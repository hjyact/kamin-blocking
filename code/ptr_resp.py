"""Pointer-helper responsibility by position type: gate, copy probability of the target and posterior responsibility
r_c = l c/((1-l) q + l c), on induction-copyable positions (bigram seen earlier) vs other positions, validation text.
python ptr_resp.py runs/X.json [...]"""
import sys, json, numpy as np, torch
src = open("circuits.py").read().split("g0 = torch.Generator()")[0]; exec(src)
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); rng = np.random.default_rng(3)
B = [torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in rng.integers(0, len(va) - 1100, 8)])).cuda() for _ in range(6)]
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    cap = {}; m.head.register_forward_hook(lambda mod, i, o: cap.__setitem__("h", i[0]))
    out = {True: [], False: []}
    with torch.no_grad():
        for X in B:
            x, y = X[:, :-1], X[:, 1:]; T = x.size(1)
            lg = m(x).float(); q = lg.log_softmax(-1).gather(-1, y[..., None]).squeeze(-1).exp()
            lam = torch.sigmoid(m.apg(cap["h"]).float()).squeeze(-1); P = g["PTR"][0]
            c = (P * (x.unsqueeze(1) == y.unsqueeze(2)).float()).sum(-1); r = lam * c / ((1 - lam) * q + lam * c).clamp_min(1e-12)
            ind = ((x.unsqueeze(1) == x.unsqueeze(2)) & (y.unsqueeze(1) == y.unsqueeze(2)) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), -1)).any(-1)
            for k in (True, False): out[k].append(torch.stack([lam[ind == k], c[ind == k], r[ind == k]]).cpu())
    for k, nm in ((True, "induction-copyable"), (False, "other")):
        a = torch.cat(out[k], 1); print(jp.split("/")[-1][:-5], f"{nm:18s} gate {a[0].mean():.3f}  copy p(target) {a[1].mean():.3f}  responsibility {a[2].mean():.3f}", flush=True)
