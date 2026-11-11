"""Tool delegation: sequences of "a+b=c;" problems (2-digit a, b; answers as 3 digit tokens). A calculator tool puts all
its mass on the correct next digit of each answer; the LM is trained alone, mixed with the tool (learned gate), or mixed
with the tool plus its own plain CE. Reported: answer-digit accuracy of the LM alone (tool removed).
python synth_tool.py NAME MODE   (MODE = alone | mix | fix)"""
import sys, json, numpy as np, torch, torch.nn.functional as F
name, mode = sys.argv[1], sys.argv[2]
DIG, PLUS, EQ, SEP = 0, 10, 11, 12            # digits 0-9, '+', '=', ';'
NPROB = 8; PL = 2 + 1 + 2 + 1 + 3 + 1; T = NPROB * PL   # 10 tokens per problem
SRC = open("./train.py").read().split('EXPERT = [None]\n')[0]
g = {"__name__": "synth"}; old = sys.argv; sys.argv = ["x", "--name", name, "--d", "256", "--layers", "4", "--heads", "4", "--seq", str(T)]
try: exec(SRC, g)
finally: sys.argv = old
SEED = int(__import__("os").environ.get("SEED", "0")); dev = "cuda"; torch.manual_seed(SEED); model = g["GPT"]().to(dev)
gate = torch.nn.Linear(256, 1).to(dev); torch.nn.init.zeros_(gate.weight); torch.nn.init.constant_(gate.bias, -3.0)
def batch(n, rng):
    x = np.zeros((n, T), np.int64); ans = np.zeros((n, T), bool)
    for b in range(n):
        seq = []; pos = []
        for _ in range(NPROB):
            a, c = rng.integers(10, 100, 2); s = a + c
            seq += [a // 10, a % 10, PLUS, c // 10, c % 10, EQ]; pos += [len(seq), len(seq) + 1, len(seq) + 2]
            seq += [s // 100, (s // 10) % 10, s % 10, SEP]
        x[b] = seq; ans[b, [p - 1 for p in pos]] = True       # position p-1 predicts answer digit at p
    xt = torch.from_numpy(x).to(dev); return xt[:, :-1], xt[:, 1:], torch.from_numpy(ans[:, :-1]).to(dev)
hidden = {}
def hook(m, i, o): hidden["h"] = i[0]
model.head.register_forward_hook(hook)
params = list(model.parameters()) + list(gate.parameters()); opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
rng = np.random.default_rng(SEED); vx, vy, va = batch(1000, np.random.default_rng(9))
STEPS = int(__import__('os').environ.get('STEPS', '4000'))
for step in range(STEPS + 1):
    x, y, ans = batch(64, rng)
    with torch.autocast("cuda", torch.bfloat16): lg = model(x)
    lp = lg.float().log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)          # LM log-prob of the target
    if mode == "sub":   # control: trained alone, each answer digit kept in the loss with probability 1-REL (only the helper's errors)
        keep = (~ans) | (torch.rand(ans.shape, device=ans.device) >= float(__import__("os").environ.get("REL", "1.0")))
        loss = -lp[keep].mean() if False else -(lp * keep).sum() / ans.numel()
    elif mode == "mask" and step < int(__import__("os").environ.get("SWITCH", "0")):   # control: same steps, answer digits excluded from the loss, no tool
        loss = -lp[~ans].mean()
    elif mode == "mask" or mode == "alone" or (int(__import__("os").environ.get("THENALONE", "0")) and step >= int(__import__("os").environ.get("SWITCH", "0"))):
        loss = -lp.mean()
    else:
        lam = torch.sigmoid(gate(g["rms"](hidden["h"]).float())).squeeze(-1) * ans.float()   # tool only speaks at answer digits
        REL = float(__import__("os").environ.get("REL", "1.0"))   # tool reliability: right on a random REL fraction of answer digits, wrong (all mass on another digit) otherwise
        right = ans & (torch.rand(ans.shape, device=ans.device) < REL)
        ltool = torch.where(right, torch.zeros_like(lp), torch.full_like(lp, -30.0))
        mix = torch.logaddexp(torch.log1p(-lam + 1e-6) + lp, torch.log(lam + 1e-9) + ltool)
        fixw = float(__import__("os").environ.get("FIXW", "1.0")); sw = int(__import__("os").environ.get("SWITCH", "0"))
        on = (mode == "fix") and step >= sw
        loss = -mix.mean() - (fixw * lp.mean() if on else 0.0)
        if mode == "jacobs":   # Jacobs et al. (1991): each component's own error weighted by its gate share, no mixture term
            loss = -((1 - lam) * lp + lam * ltool).mean()
        if mode == "jacobsmix":   # mixture likelihood plus the LM's own error weighted by its gate share (1 - lambda)
            loss = -mix.mean() - ((1 - lam) * lp).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % int(__import__('os').environ.get('EVERY', '500')) == 0:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): pv = torch.cat([model(vx[i:i + 200]).argmax(-1) for i in range(0, len(vx), 200)])
        acc = (pv[va] == vy[va]).float().mean().item(); print(name, step, round(loss.item(), 4), "LM-alone answer acc", round(acc, 4), flush=True)
print("FINAL", name, round(acc, 4))
