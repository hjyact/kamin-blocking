"""Exact Price decomposition of the change in mixture log-likelihood between logged steps:
d log p = [(1-lam_bar) dq + lam_bar dc + (c-q)_bar dlam] / L(p, p'), L = logarithmic mean.
Reports cumulative LM transmission, copy transmission and selection, overall and split by whether the target was already in
the context (c>0 at the end) .  python price.py runs/X_s0_price.npz [...]"""
import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f); st, g = d["steps"], d["g"].astype(np.float64)   # g: [n, 3, B, T] = lam, q, c
    lam, q, c = g[:, 0], g[:, 1], g[:, 2]; p = (1 - lam) * q + lam * c
    seen = c[-1] > 1e-6
    acc = {k: np.zeros(3) for k in ("all", "seen", "new")}
    for k in range(len(st) - 1):
        l0, l1, q0, q1, c0, c1, p0, p1 = lam[k], lam[k + 1], q[k], q[k + 1], c[k], c[k + 1], p[k], p[k + 1]
        lb = (l0 + l1) / 2; dqm = ((c0 - q0) + (c1 - q1)) / 2
        L = np.where(np.abs(p1 - p0) > 1e-12, (p1 - p0) / (np.log(p1 + 1e-30) - np.log(p0 + 1e-30)), p0)
        terms = np.stack([(1 - lb) * (q1 - q0), lb * (c1 - c0), dqm * (l1 - l0)]) / np.maximum(L, 1e-30)
        for key, m in (("all", np.ones_like(seen)), ("seen", seen), ("new", ~seen)): acc[key] += terms[:, m].mean(1)
    tot = np.log(p[-1] + 1e-30) - np.log(p[0] + 1e-30)
    print(f.split("/")[-1], "steps", st[0], "->", st[-1])
    for key, m in (("all", np.ones_like(seen)), ("seen", seen), ("new", ~seen)):
        a = acc[key]; print(f"  {key:4s} (frac {m.mean():.2f}): LM transmission {a[0]:+.3f}  copy transmission {a[1]:+.3f}  selection {a[2]:+.3f}  sum {a.sum():+.3f}  check {tot[m].mean():+.3f}")
