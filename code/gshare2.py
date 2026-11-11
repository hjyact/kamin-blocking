import sys, json
L = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{") and '"gates"' in l]
labs = [f"{o} p{p} r{r}" for o in ("Adam+noise", "Adam", "SGD+noise") for p in (1, 2, 3) for r in (1, 2, 4)]
rows = {r["step"]: r["gates"] for r in L}; steps = [s for s in (300, 400, 600, 1000, 2000, 3050) if s in rows]
print(" " * 18 + " ".join(f"{s:>6d}" for s in steps))
for k, lab in enumerate(labs): print(f"{lab:18s}" + " ".join(f"{rows[s][k][1] / (rows[s][k][0] + rows[s][k][1]):6.3f}" for s in steps))
