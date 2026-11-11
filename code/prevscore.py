# attention of one head to the previous position on natural text, at given checkpoints
import sys, os, torch
exec(open("./headrace.py").read().split("final = sys.argv[-1]")[0])
L, H = (int(v) for v in os.environ["HEAD"].split(",")); K = int(os.environ.get("OFF", "1"))
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval(); tot = 0
    for X in Xs[:2]:
        g["ATTN"][0] = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): m(X[:, :-1])
        a = g["ATTN"][0][L][:, H].float(); g["ATTN"][0] = None; T = a.size(-1); t = torch.arange(K, T, device=a.device)
        tot += a[:, t, t - K].mean().item() / 2
    print(ck, f"L{L}H{H} offset attention {tot:.3f}")
