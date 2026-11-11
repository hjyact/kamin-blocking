# B share gB/(gA+gB) per gate pair over training, from a --gate_cfg log
import sys, json
L = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{") and '"gates"' in l]
cfg = [c for c in json.dumps(0) and []]
steps = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "300,400,600,1000,2000,3050").split(",")]
labels = [f"p{p} r{r}" for p in (1, 2, 3) for r in (1, 1.5, 2, 4)] + ["p3 r2 Adam", "p3 r4 Adam", "p3 r1 Adam"]
rows = {r["step"]: r["gates"] for r in L}
print("pair        " + " ".join(f"{s:>6d}" for s in steps if s in rows) + "   gA,gB at end")
last = L[-1]["gates"]
for k, lab in enumerate(labels):
    print(f"{lab:12s}" + " ".join(f"{rows[s][k][1] / (rows[s][k][0] + rows[s][k][1]):6.3f}" for s in steps if s in rows) + f"   {last[k][0]:.3f},{last[k][1]:.3f}")
