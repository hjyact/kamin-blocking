"""Per-head previous-token and induction attention scores on a repeated random sequence (tokens < 5000, 2 x 256).
python circuits.py runs/X.json [...]"""
import sys, json, torch
SRC = open("./train.py").read().split('D = "./data"')[0]
def build(cfg):
    argv = ["x"]
    for k, v in cfg.items():
        if k == "out" or (isinstance(v, bool) and k.startswith("cache")): continue   # LM head alone: drop the copy path
        if isinstance(v, bool):
            if v: argv.append("--" + k)
        else: argv += ["--" + k, str(v)]
    g = {"__name__": "circ"}; old = sys.argv; sys.argv = argv
    try: exec(SRC, g)
    finally: sys.argv = old
    return g
g0 = torch.Generator().manual_seed(4242); H = 256; half = torch.randint(0, 5000, (8, H), generator=g0); X = torch.cat([half, half], 1).cuda()
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    g["ATTN"][0] = []
    with torch.no_grad():
        lg = m(X).float()[:, :-1]; nl = -lg.log_softmax(-1).gather(-1, X[:, 1:, None]).squeeze(-1); gain = (nl[:, :H - 1].mean() - nl[:, H:].mean()).item()
    t = torch.arange(H + 1, 2 * H); out = []
    for li, a in enumerate(g["ATTN"][0]):           # a: B,h,T,T
        prev = a[:, :, torch.arange(1, 2 * H), torch.arange(0, 2 * H - 1)].mean((0, 2))      # attend t -> t-1
        ind = a[:, :, t, t - H + 1].mean((0, 2))                                                  # attend t -> token after previous occurrence
        out.append((li, prev, ind))
    name = jp.split("/")[-1][:-5]
    bp = max((p.max().item(), li) for li, p, _ in out); bi = max((i.max().item(), li) for li, _, i in out)
    print(name, "best prev-token head", round(bp[0], 3), "layer", bp[1], "| best induction head", round(bi[0], 3), "layer", bi[1],
          "| heads with prev>0.5:", sum(int((p > 0.5).sum()) for _, p, _ in out), "induction>0.3:", sum(int((i > 0.3).sum()) for _, _, i in out), "LM-head copy gain", round(gain, 3), flush=True)
    print("  per-layer max prev", [round(p.max().item(), 2) for _, p, _ in out], "max ind", [round(i.max().item(), 2) for _, _, i in out], flush=True)
