"""Backward blocking / release from overshadowing under curvature-aware learning. Per-key logit tables: LM theta, helper phi
(N keys x K labels), combined by a logit sum. Phase 1: joint training on all keys. Phase 2 on keys S only: helper alone is
trained toward the right label (A+, backward blocking) or toward uniform (A-, release). LM is never in the phase-2 loss.
Learners: 'kalman' (online Laplace / natural gradient with accumulated full Fisher, incl. LM-helper cross terms), 'diag'
(same but diagonal), 'sgd'. Reports LM-alone mean log q_y on S and on control keys before/after phase 2.
python synth_bb.py LEARNER PHASE2(plus|minus)"""
import sys, torch
SEED = int(__import__('os').environ.get('SEED', '0')); torch.manual_seed(SEED); L, P2 = sys.argv[1], sys.argv[2]; N, K = 16, 8; D = 2 * N * K; T1, T2, LR = 3000, 3000, 0.5
y = torch.randint(0, K, (N,)); w = torch.zeros(D, dtype=torch.float64); Lam = torch.eye(D, dtype=torch.float64) * 1.0
th = lambda w, i: w[i * K:(i + 1) * K]; ph = lambda w, i: w[N * K + i * K: N * K + (i + 1) * K]
S = torch.arange(N // 2)          # keys revalued in phase 2; the rest are controls
def step(i, target, lm_in):
    global w, Lam
    z = ph(w, i) + (th(w, i) if lm_in else 0); p = z.softmax(0); gz = p - target; Hz = torch.diag(p) - torch.outer(p, p)
    idx = [torch.arange(N * K + i * K, N * K + (i + 1) * K)] + ([torch.arange(i * K, (i + 1) * K)] if lm_in else [])
    J = torch.zeros(K, D, dtype=torch.float64)
    for ix in idx: J[torch.arange(K), ix] = 1.0
    g = J.T @ gz
    if L == "sgd": w = w - LR * g; return
    F = J.T @ Hz @ J
    if L == "kalman": Lam = Lam + F; w = w - torch.linalg.solve(Lam, g)
    else: Lam = Lam + torch.diag(torch.diag(F)); w = w - g / torch.diag(Lam)
def lmq(keys): return torch.stack([th(w, i).log_softmax(0)[y[i]] for i in keys]).mean().item()
oh = lambda i: torch.nn.functional.one_hot(y[i], K).double()
for t in range(T1): i = int(torch.randint(0, N, ())); step(i, oh(i), True)
ctl = torch.arange(N // 2, N); b_s, b_c = lmq(S), lmq(ctl)
for t in range(T2):
    i = int(S[torch.randint(0, len(S), ())]); step(i, oh(i) if P2 == "plus" else torch.full((K,), 1.0 / K, dtype=torch.float64), False)
a_s, a_c = lmq(S), lmq(ctl)
print(f"{L:6s} {P2:5s} LM log q_y  revalued {b_s:+.3f} -> {a_s:+.3f} (d {a_s-b_s:+.3f})   control {b_c:+.3f} -> {a_c:+.3f} (d {a_c-b_c:+.3f})")
