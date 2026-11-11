# head-start amplification: log(odds_r / odds_1) / log(r), odds = gA/gB, relative to the r=1 pair of the same parametrisation
import sys, json, math
L = {json.loads(l)["step"]: json.loads(l)["gates"] for l in open(sys.argv[1]) if l.startswith("{") and '"gates"' in l}
grp = {"p1 SGD": (0, [1, 1.5, 2, 4]), "p2 SGD": (4, [1, 1.5, 2, 4]), "p3 SGD": (8, [1, 1.5, 2, 4])}
for st in [400, 600, 800, 1000, 2000, 3050]:
    out = []
    for name, (b, rs) in grp.items():
        o1 = L[st][b][0] / L[st][b][1]
        out.append(name + " " + "/".join(f"{math.log((L[st][b + j][0] / L[st][b + j][1]) / o1) / math.log(r):.2f}" for j, r in enumerate(rs) if r > 1))
    o1 = L[st][14][0] / L[st][14][1]; out.append("p3 Adam " + "/".join(f"{math.log((L[st][12 + j][0] / L[st][12 + j][1]) / o1) / math.log(r):.2f}" for j, r in enumerate([2, 4])))
    print(st, " | ".join(out))
