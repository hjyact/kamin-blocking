import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
# (a) takeover time vs 1/(1-rho)
x = np.array([2, 5, 10]); t = np.array([750, 1750, 3250])
ax[0].plot(x, t, "o", c="#1f77b4"); xx = np.linspace(0, 21, 50); ax[0].plot(xx, 350 * xx, "--", c="#999", lw=0.8, label="350/(1-ρ)")
ax[0].annotate("", xy=(20, 4000), xytext=(20, 3000), arrowprops=dict(arrowstyle="->", color="#d62728")); ax[0].text(13.5, 2600, "ρ=0.95: >4000", fontsize=7, color="#d62728")
ax[0].set_xlabel("1/(1-ρ)  (helper reliability ρ)"); ax[0].set_ylabel("steps until the LM takes over"); ax[0].set_title("(a) takeover law", fontsize=9); ax[0].legend(frameon=False, fontsize=7)
# (b) replicator equilibrium of released gates: mean copy-head responsibility per gate bin (Table equil)
gate = [.003, .026, .072, .142, .245, .347, .447, .548, .648, .749, .851, .926, .973, .996]
R = {"135M": [.003, .027, .074, .146, .247, .355, .448, .550, .648, .756, .859, .922, .974, .998],
     "360M": [.003, .026, .072, .145, .249, .355, .453, .539, .652, .748, .847, .929, .974, .998],
     "1.7B": [.003, .025, .070, .144, .250, .346, .441, .555, .649, .746, .856, .924, .974, .998]}
ax[1].plot([0, 1], [0, 1], "--", c="#999", lw=0.8, label="equilibrium")
for (n, r), c, mk in zip(R.items(), ["#9ecae1", "#4292c6", "#08306b"], ["o", "s", "^"]): ax[1].plot(gate, r, mk, c=c, ms=3.5, label=n)
ax[1].set_xlabel("effective gate λ"); ax[1].set_ylabel("helper responsibility"); ax[1].set_title("(b) gates at equilibrium (released)", fontsize=9)
ax[1].legend(frameon=False, fontsize=6, loc="upper left")
# (c) graded blocking across sizes
lab = ["0", "<.1", ".1-.5", ".5-.9", ".9-.99", "≥.99"]
for v, c, n in [([-0.047, .065, .342, .616, .723, .791], "#9467bd", "135M"), ([-.052, .050, .273, .467, .558, .504], "#2ca02c", "360M"), ([-.057, .034, .206, .268, .321, .327], "#1f77b4", "1.7B")]:
    ax[2].plot(range(6), v, "o-", c=c, label=n)
ax[2].axhline(0, c="#999", lw=0.6); ax[2].set_xticks(range(6)); ax[2].set_xticklabels(lab, fontsize=7, rotation=30)
ax[2].set_xlabel("copy head's p(target)"); ax[2].set_ylabel("LM alone − vanilla (nats)"); ax[2].set_title("(c) graded blocking (released)", fontsize=9); ax[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("fig_theory.pdf"); fig.savefig("fig_theory.png", dpi=160)
