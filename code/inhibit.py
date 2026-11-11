"""Conditioned inhibition: does a product-trained LM head push down tokens that occur in the context but are not the target?
Index = mean log q over distinct in-context non-target tokens minus mean log q over tokens absent from the context, at
positions t>=512 of validation sequences, LM head alone.  python inhibit.py runs/X.json [...]"""
import sys, json, numpy as np, torch
src = open("circuits.py").read().split("g0 = torch.Generator()")[0]; exec(src)
va = np.memmap("./data/val.bin", dtype=np.uint16, mode="r"); rng = np.random.default_rng(5)
starts = rng.integers(0, len(va) - 1100, 32); X = torch.from_numpy(np.stack([va[s:s + 1025].astype(np.int64) for s in starts])).cuda()
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    ins, outs, tg = [], [], []
    with torch.no_grad():
        for b in range(0, 32, 8):
            x = X[b:b + 8, :-1]; y = X[b:b + 8, 1:]; lq = m(x).float().log_softmax(-1)
            for i in range(8):
                for t in range(512, 1024, 16):
                    ctx = torch.unique(x[i, :t + 1]); ctx = ctx[ctx != y[i, t]]
                    mask = torch.ones(lq.size(-1), dtype=torch.bool, device=x.device); mask[ctx] = False; mask[y[i, t]] = False
                    ins.append(lq[i, t, ctx].mean().item()); outs.append(lq[i, t][mask].mean().item()); tg.append(lq[i, t, y[i, t]].item())
    print(jp.split("/")[-1][:-5], f"in-context non-target {np.mean(ins):.3f}  absent {np.mean(outs):.3f}  index {np.mean(ins) - np.mean(outs):.3f}  target {np.mean(tg):.3f}", flush=True)
