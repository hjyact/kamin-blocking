"""Per-token val loss split by whether the target already occurred in the context (seen) or not (new), for the full model
and for the LM head alone (cache path removed).  python copyprobe.py runs/X.json [...]"""
import sys, json, numpy as np, torch
sys.argv, files = sys.argv[:1], sys.argv[1:]
exec(open("./lmonly.py").read().split("@torch.no_grad()")[0])
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r")
@torch.no_grad()
def split(m, T, cache, nb=150):
    S = {"seen": [0.0, 0], "new": [0.0, 0]}
    for i in range(nb):
        o = i * T * 2; x = torch.from_numpy(np.stack([va[o + b * T:o + b * T + T + 1].astype(np.int64) for b in range(2)])).cuda()
        inp, tgt = x[:, :-1], x[:, 1:]
        with torch.autocast("cuda", torch.bfloat16): out = m(inp)
        lp = out.float() if cache else out.float().log_softmax(-1)
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        seen = ((inp.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool))).any(-1)
        for k, msk in (("seen", seen), ("new", ~seen)): S[k][0] += nll[msk].sum().item(); S[k][1] += msk.sum().item()
    return {k: round(v[0] / v[1], 4) for k, v in S.items()}
for jp in files:
    cfg = json.load(open(jp))["cfg"]; sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); sd = {a: b.float() if b.is_floating_point() else b for a, b in sd.items()}
    r = {}
    for strip in (False, True):
        m = build(cfg, strip); m.load_state_dict(sd, strict=False); m.eval(); r["lm" if strip else "full"] = split(m, cfg["seq"], cfg.get("cachelm") and not strip)
    print(jp.split("/")[-1][:-5], r, flush=True)
