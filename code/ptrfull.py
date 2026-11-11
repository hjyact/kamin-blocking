"""Full mixture (LM + pointer helper) vs LM alone on validation for --attnptr checkpoints.  python ptrfull.py runs/X.json [...]"""
import sys, json, numpy as np, torch
SRC = open("./train.py").read().split('D = "./data"')[0]
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(1234)
B = [np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 8)]) for _ in range(16)]
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; argv = ["x"]
    for k, v in cfg.items():
        if k in ("out", "lmaux", "ptr_until"): continue   # evaluation: the plain mixture loss
        if isinstance(v, bool):
            if v: argv.append("--" + k)
        else: argv += ["--" + k, str(v)]
    g = {"__name__": "pf"}; old = sys.argv; sys.argv = argv; exec(SRC, g); sys.argv = old; g["OSTAT"] = [None]
    m = g["GPT"]().cuda(); sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False)
    full, alone = [], []
    with torch.no_grad():
        for b in B:
            x = torch.from_numpy(b).cuda()
            with torch.autocast("cuda", torch.bfloat16):
                m.train(); full.append(m(x[:, :-1], x[:, 1:]).item()); m.eval(); alone.append(m(x[:, :-1], x[:, 1:]).item())
    print(jp.split("/")[-1][:-5], f"full mixture {np.mean(full):.4f}  LM alone {np.mean(alone):.4f}", flush=True)
