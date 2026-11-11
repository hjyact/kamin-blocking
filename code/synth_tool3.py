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
    x = np.zeros((n, T), np.int64); ans = np.zeros((n, T), bool); cov = np.zeros((n, T), bool)
    for b in range(n):
        seq = []; pos = []; cpos = []
        for _ in range(NPROB):
            a, c = rng.integers(10, 100, 2); s = a + c
            seq += [a // 10, a % 10, PLUS, c // 10, c % 10, EQ]; pos += [len(seq), len(seq) + 1, len(seq) + 2]
            if (a % 10) + (c % 10) < 10: cpos += pos[-3:]   # tool covers only problems without a units carry
            covd = (a % 10) + (c % 10) < 10; _o = __import__('os').environ
            off = 20 if (_o.get('DISJ') and (covd or (_o.get('DISJMIX') and rng.random() < 0.5))) else 0   # covered answers on their own ids; DISJMIX: half of the uncovered too
            seq += [off + s // 100, off + (s // 10) % 10, off + s % 10, SEP]
        x[b] = seq; ans[b, [p - 1 for p in pos]] = True; cov[b, [p - 1 for p in cpos]] = True       # position p-1 predicts answer digit at p
    xt = torch.from_numpy(x).to(dev); return xt[:, :-1], xt[:, 1:], torch.from_numpy(ans[:, :-1]).to(dev), torch.from_numpy(cov[:, :-1]).to(dev)
hidden = {}
def hook(m, i, o): hidden["h"] = i[0]
model.head.register_forward_hook(hook)
params = list(model.parameters()) + list(gate.parameters()); opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
rng = np.random.default_rng(SEED); vx, vy, va, vc = batch(1000, np.random.default_rng(9))
STEPS = int(__import__('os').environ.get('STEPS', '4000'))
for step in range(STEPS + 1):
    x, y, ans, cov = batch(64, rng)
    with torch.autocast("cuda", torch.bfloat16): lg = model(x)
    lp = lg.float().log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)          # LM log-prob of the target
    PRE = int(__import__("os").environ.get("PRE", "0"))
    if step == PRE and __import__("os").environ.get("EPS2"):   # Adam eps changed only once the tool is attached
        for pg in opt.param_groups: pg["eps"] = float(__import__("os").environ["EPS2"])
    if step == PRE and __import__("os").environ.get("LR2"):   # control: only the learning rate lowered once the tool is attached
        for pg in opt.param_groups: pg["lr"] = float(__import__("os").environ["LR2"])
    if mode == "mask":   # no tool: covered answer positions simply removed from the loss (Proposition: blocking = complement training)
        loss = -(lp * (~cov).float()).sum() / (~cov).float().sum()
    elif mode == "alone" or step < PRE:
        loss = -lp.mean()
    else:
        lam = torch.sigmoid(gate(g["rms"](hidden["h"]).float())).squeeze(-1) * cov.float()   # tool only speaks where it covers
        REL = float(__import__("os").environ.get("REL", "1.0"))   # helper reliability on the positions it covers
        right = cov & (torch.rand(cov.shape, device=cov.device) < REL)
        ltool = torch.where(right, torch.zeros_like(lp), torch.full_like(lp, -30.0))
        mix = torch.logaddexp(torch.log1p(-lam + 1e-6) + lp, torch.log(lam + 1e-9) + ltool)
        fixw = float(__import__("os").environ.get("FIXW", "1.0")); sw = int(__import__("os").environ.get("SWITCH", "0"))
        on = (mode == "fix") and step >= sw
        loss = -mix.mean() - (fixw * lp.mean() if on else 0.0)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if mode not in ("alone", "mask") and step >= PRE and step % 50 == 0:   # gate mass on covered answer digits (training batch) vs LM's own prob of the target there
        with torch.no_grad(): print("GATE", step, round(lam[cov].mean().item(), 4), round(lp[cov].exp().mean().item(), 4), flush=True)
    if step % int(__import__('os').environ.get('EVERY', '250')) == 0:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): pv = torch.cat([model(vx[i:i + 200]).argmax(-1) for i in range(0, len(vx), 200)])
        ok = (pv == vy); acc = ok[va].float().mean().item(); a_cov = ok[va & vc].float().mean().item(); a_unc = ok[va & ~vc].float().mean().item()
        print(name, step, round(loss.item(), 4), "acc", round(acc, 4), "tool-covered", round(a_cov, 4), "uncovered", round(a_unc, 4), flush=True)
print("FINAL", name, round(acc, 4), round(a_cov, 4), round(a_unc, 4))
