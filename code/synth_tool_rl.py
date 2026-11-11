"""Tool delegation under reinforcement learning. Contextual bandit at every answer digit of "a+b=c;" problems
(teacher-forced context). Actions: the 10 digits or CALL. CALL returns the right digit at cost c (reward 1-c);
a self-emitted digit earns 1 if right. REINFORCE with a running baseline per action family.
MODE = rl (plain) | cf (plain + counterfactual self-rollout: a digit is also sampled from the digit-only policy at
every position and trained on its own reward, whether or not CALL was taken) | alone (no CALL action).
Reported: LM-alone accuracy (argmax over digits), CALL rate, mean reward."""
import sys, os, numpy as np, torch
name, mode = sys.argv[1], sys.argv[2]
COST = float(os.environ.get("COST", "0.2")); SEED = int(os.environ.get("SEED", "0")); STEPS = int(os.environ.get("STEPS", "6000"))
PLUS, EQ, SEP, CALL = 10, 11, 12, 13
NPROB = 8; T = NPROB * 10
SRC = open("./train.py").read().split('EXPERT = [None]\n')[0]
g = {"__name__": "synth"}; old = sys.argv; sys.argv = ["x", "--name", name, "--d", "256", "--layers", "4", "--heads", "4", "--seq", str(T)]
try: exec(SRC, g)
finally: sys.argv = old
dev = "cuda"; torch.manual_seed(SEED); model = g["GPT"]().to(dev)
def batch(n, rng):
    x = np.zeros((n, T), np.int64); ans = np.zeros((n, T), bool)
    for b in range(n):
        seq = []; pos = []
        for _ in range(NPROB):
            a, c = rng.integers(10, 100, 2); s = a + c
            seq += [a // 10, a % 10, PLUS, c // 10, c % 10, EQ]; pos += [len(seq), len(seq) + 1, len(seq) + 2]
            seq += [s // 100, (s // 10) % 10, s % 10, SEP]
        x[b] = seq; ans[b, [p - 1 for p in pos]] = True
    xt = torch.from_numpy(x).to(dev); return xt[:, :-1], xt[:, 1:], torch.from_numpy(ans[:, :-1]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
rng = np.random.default_rng(SEED); vx, vy, va = batch(1000, np.random.default_rng(9))
acts = list(range(10)) + ([CALL] if mode != "alone" else [])
base_full = 0.0; base_self = 0.0
for step in range(STEPS + 1):
    x, y, ans = batch(64, rng)
    with torch.autocast("cuda", torch.bfloat16): lg = model(x)
    lg = lg[ans][:, :14].float(); yt = y[ans]                                  # N answer positions
    lp = lg[:, acts].log_softmax(-1)
    pr = lp.exp()                                                      # exact expected reward (no sampling noise)
    if mode != "alone": J = pr.gather(-1, yt[:, None]).squeeze(-1) + pr[:, 10] * (1 - COST)
    else: J = pr.gather(-1, yt[:, None]).squeeze(-1)
    loss = -J.mean(); r = J
    if mode == "cf":                                                   # counterfactual self-objective on the digit-only policy
        loss = loss - lg[:, :10].softmax(-1).gather(-1, yt[:, None]).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 500 == 0:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            L = torch.cat([model(vx[i:i + 50])[va[i:i + 50]][:, :14].float() for i in range(0, len(vx), 50)])
        acc = (L[:, :10].argmax(-1) == vy[va]).float().mean().item()
        callr = (L[:, acts].softmax(-1)[:, -1].mean().item()) if mode != "alone" else 0.0
        print(name, step, "LM-alone acc", round(acc, 4), "P(call)", round(callr, 4), "reward", round(r.mean().item(), 4), flush=True)
print("FINAL", name, round(acc, 4), round(callr, 4))
