# best induction score among heads other than the excluded ones, and the sum over all other heads
import sys, os, numpy as np, torch
exec(open("./headrace.py").read().split("final = sys.argv[-1]")[0])
for arg in sys.argv[1:]:
    ck, ex = (arg.split("@") + [""])[:2]; excl = [tuple(int(v) for v in e.split(",")) for e in ex.split(";") if e]
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval(); S = scores(False).clone()
    for e in excl: S[e] = -1
    v = S.flatten(); H = S.size(1); top = torch.topk(v, 3).indices.tolist()
    print(f"{ck} excl {excl}: best other {v.max():.3f} ({'L%dH%d' % divmod(top[0], H)}), 2nd {v[top[1]]:.3f}, 3rd {v[top[2]]:.3f}, sum of others >0.05: {v[v > 0.05].sum():.3f}")
