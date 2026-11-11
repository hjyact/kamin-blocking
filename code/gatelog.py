# gate / LM / copy trajectories from --pricelog files: mean gate, gate on positions whose target is in the context (copy can help)
# and elsewhere, mean LM prob and copy prob of the target, mixture loss
import sys, numpy as np
for n in sys.argv[1:]:
    z = np.load(f"runs/{n}_s0_price.npz"); st, G = z["steps"], z["g"]   # G: steps x 3 (gate, q_y, c_y) x B x T
    print(n)
    for i, s in enumerate(st):
        lam, q, c = G[i, 0], G[i, 1], G[i, 2]; seen = c > 1e-6; mix = (1 - lam) * q + lam * c
        print(f"  step {s:5d} gate {lam.mean():.3f} (copyable-ish {lam[seen].mean():.3f}, other {lam[~seen].mean():.3f})  q_y {q.mean():.3f} (seen {q[seen].mean():.3f})  c_y|seen {c[seen].mean():.3f}  mix loss {-np.log(mix + 1e-9).mean():.3f}  LM-alone loss {-np.log(q + 1e-9).mean():.3f}")
