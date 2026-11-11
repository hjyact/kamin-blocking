import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(9.4, 2.35))
# (a) cost of withholding all copyable credit (f=1) for a window: onset delay vs window, baseline 1300 +- 90 (three identical runs)
lab = [".0-.3", ".3-.6", ".6-.9", ".9-1.2", "1.2-1.5", "0-1", ".7-1.7"]; d = [-176, 191, 200, 198, 294, 785, 1696]; L = [300] * 5 + [1000, 1000]
x = np.arange(len(d)); ax[0].bar(x, d, color=["#999"] * 5 + ["#1f77b4", "#d62728"]); ax[0].scatter(x, L, marker="_", s=120, c="k", zorder=3, label="window length")
ax[0].axhspan(-90, 90, color="#ddd", zorder=0); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=90, fontsize=6); ax[0].set_xlabel("window (k steps)", fontsize=7)
ax[0].set_ylabel("onset delay (steps)"); ax[0].set_title("(a) when credit is withheld", fontsize=9); ax[0].legend(frameon=False, fontsize=6, loc="upper left")
# (b) hysteresis: induction head at the end of training against the blocked fraction f, blocked from the start or from step 1600
fb = [0, 0.15, 0.3, 0.5, 1.0]; hb = [0.16, 0.113, 0.078, 0.056, 0.006]; fa = [0.5, 0.75, 0.9, 1.0]; ha = [0.109, 0.087, 0.053, 0.052]
ax[1].plot(fb, hb, "o-", c="#444", ms=3, label="blocked from the start"); ax[1].plot(fa, ha, "s-", c="#d62728", ms=3, label="blocked after the onset")
ax[1].set_xlabel("fraction of copyable credit blocked, $f$"); ax[1].set_ylabel("induction head, end"); ax[1].set_title("(b) keeping is cheaper than forming", fontsize=9); ax[1].legend(frameon=False, fontsize=6)
# (c) blocking between two heads: final score of the head that came second / was not pretrained, with and without the other head first
grp = ["attn.\nsup.", "pair 1\nA", "pair 1\nB", "pair 2\nC", "pair 2\nD"]; alone = [0.317, 0.299, 0.122, 0.312, 0.261]; after = [0.254, 0.023, 0.027, 0.037, 0.070]
x = np.arange(len(grp)); w = 0.38
ax[2].bar(x - w / 2, alone, w, color="#444", label="without the other first"); ax[2].bar(x + w / 2, after, w, color="#d62728", label="other head formed first")
ax[2].set_xticks(x); ax[2].set_xticklabels(grp, fontsize=6); ax[2].set_ylabel("induction score, end"); ax[2].set_title("(c) blocking between heads", fontsize=9); ax[2].set_ylim(0, 0.55); ax[2].legend(frameon=False, fontsize=5.5, loc="upper right", ncol=1); ax[2].axvline(0.5, c="#bbb", lw=0.6, ls=":"); ax[2].text(2.5, 0.36, "shared error", ha="center", fontsize=6, color="#555")
# (d) depth in isolation: two gates on the same head (identical cues), head start 2:1; second gate's share over training
import json
L = [json.loads(l) for l in open("./logs/GSAME_s0.log") if l.startswith("{") and '"gates"' in l]
st = [r["step"] - 300 for r in L if r["step"] >= 300]; G = [r["gates"] for r in L if r["step"] >= 300]
for k, lab, c, ls in [(2, "$p=1$", "#999", "-"), (6, "$p=2$", "#444", "-"), (10, "$p=3$", "#d62728", "-"), (12, "$p=3$, Adam", "#1f77b4", "--")]:
    ax[3].plot(st, [g[k][1] / (g[k][0] + g[k][1]) for g in G], ls, c=c, lw=1.2, label=lab)
ax[3].axhline(1 / 3, c="#bbb", lw=0.6, ls=":", zorder=0); ax[3].set_xlabel("steps of shared readout"); ax[3].set_ylabel("later gate's share"); ax[3].set_ylim(0.15, 0.64)
ax[3].set_title("(d) depth decides the race", fontsize=9); ax[3].legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
plt.tight_layout(); plt.savefig("fig_circuits.pdf"); print("ok")
