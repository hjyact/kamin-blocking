"""Redundancy of the induction circuit in our 77M models: LM-head copying gain on a repeated random sequence (copy path
stripped), intact, with the strongest induction head zeroed, and with the two strongest zeroed; top induction scores listed.
python ablate77.py runs/X.json [...]"""
import sys, json, torch
src = open("circuits.py").read().split("for jp in sys.argv[1:]:")[0]; exec(src)
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    g["ATTN"][0] = []
    with torch.no_grad(): m(X)
    t = torch.arange(H + 1, 2 * H); sc = torch.stack([a[:, :, t, t - H + 1].mean((0, 2)) for a in g["ATTN"][0]]); g["ATTN"][0] = None
    nh = sc.size(1); order = sc.flatten().argsort(descending=True).tolist(); top = [divmod(i, nh) for i in order[:3]]
    def gain(abl):
        hooks = []
        for L, h in abl:
            hd = m.blocks[L].attn.hd
            def pre(mod, args, hh=h, hd=hd):
                x = args[0].clone(); x[..., hh * hd:(hh + 1) * hd] = 0; return (x,)
            hooks.append(m.blocks[L].attn.o.register_forward_pre_hook(pre))
        with torch.no_grad(): lg = m(X).float()[:, :-1]
        for k in hooks: k.remove()
        nl = -lg.log_softmax(-1).gather(-1, X[:, 1:, None]).squeeze(-1); return (nl[:, :H - 1].mean() - nl[:, H:].mean()).item()
    print(jp.split("/")[-1][:-5], "top induction", [(L, h, round(sc[L, h].item(), 3)) for L, h in top],
          f"| copy gain intact {gain([]):.3f}  -top1 {gain(top[:1]):.3f}  -top2 {gain(top[:2]):.3f}  -top3 {gain(top[:3]):.3f}", flush=True)
