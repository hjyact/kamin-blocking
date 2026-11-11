"""Each coalition on validation: full mixture NLL, LM head alone, and copy head alone on positions whose target is in the
context.  python cpalone.py runs/X.json [...]"""
import sys, json, numpy as np, torch
SRC = open("./train.py").read().split('D = "./data"')[0]
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); r = np.random.default_rng(1234)
B = [np.stack([va[s:s + 1025].astype(np.int64) for s in r.integers(0, len(va) - 1100, 8)]) for _ in range(16)]
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; argv = ["x"]
    for k, v in cfg.items():
        if k == "out": continue
        if isinstance(v, bool):
            if v: argv.append("--" + k)
        else: argv += ["--" + k, str(v)]
    g = {"__name__": "cp"}; old = sys.argv; sys.argv = argv; exec(SRC, g); sys.argv = old
    m = g["GPT"]().cuda(); sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    g["CPLOG"][0] = []
    with torch.no_grad():
        for b in B:
            x = torch.from_numpy(b).cuda()
            with torch.autocast("cuda", torch.bfloat16): m(x[:, :-1], x[:, 1:])
    lc = torch.cat([a[0].flatten() for a in g["CPLOG"][0]]); sn = torch.cat([a[1].flatten() for a in g["CPLOG"][0]]) > 0
    lq = torch.cat([a[2].flatten() for a in g["CPLOG"][0]]); mx = torch.cat([a[3].flatten() for a in g["CPLOG"][0]])
    print(jp.split("/")[-1][:-5], f"full {-mx.mean():.4f}  LM alone {-lq.mean():.4f} (seen {-lq[sn].mean():.4f})  copy alone on seen {-lc[sn].clamp(min=-20).mean():.4f}  mixture on seen {-mx[sn].mean():.4f}  seen frac {sn.float().mean():.3f}", flush=True)
