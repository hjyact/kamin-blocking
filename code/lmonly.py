"""Evaluate a checkpoint with and without its cache path: val loss of the plain LM head vs the full model.
python lmonly.py runs/X.json [...]"""
import sys, json, numpy as np, torch
SRC = open("./train.py").read().split('D = "./data"')[0]
def build(cfg, strip):
    argv = ["x"]
    for k, v in cfg.items():
        if k in ("out",) or (strip and k == "expert"): continue
        if isinstance(v, bool):
            if v and not (strip and k.startswith("cache")): argv.append("--" + k)
        else: argv += ["--" + k, str(v)]
    g = {"__name__": "lmonly"}; old = sys.argv; sys.argv = argv
    try: exec(SRC, g)
    finally: sys.argv = old
    return g["GPT"]().cuda()
@torch.no_grad()
def val(m, T, n=4_000_000):
    va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); tot = 0.0; k = 0
    for i in range(0, min(n, len(va) - T - 1), T * 8):
        xs = np.stack([va[i + b * T:i + b * T + T + 1].astype(np.int64) for b in range(8)]); x = torch.from_numpy(xs).cuda()
        with torch.autocast("cuda", torch.bfloat16): tot += m(x[:, :-1], x[:, 1:]).item()
        k += 1
    return tot / k
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); sd = {a: b.float() if b.is_floating_point() else b for a, b in sd.items()}
    r = {}
    for strip in (False, True):
        m = build(cfg, strip); m.load_state_dict(sd, strict=False); m.eval(); r["lm_only" if strip else "full"] = round(val(m, cfg["seq"]), 4)
    print(jp.split("/")[-1][:-5], r, flush=True)
