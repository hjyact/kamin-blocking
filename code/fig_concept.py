import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 6})
fig = plt.figure(figsize=(3.3, 1.95)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
def box(x, y, w, h, t, fc, ec="#333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=0.7)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center")
def arr(x0, y0, x1, y1, c="#333"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
box(0.1, 4.0, 2.9, 1.15, "helper $c$\ncopy head, tool, expert,\nretrieved passage", "#dbe8f5")
box(0.1, 2.3, 2.9, 0.8, "language model $q$", "#f3dcdc")
box(3.7, 3.25, 3.0, 0.9, "$p=(1-\\lambda)\\,q+\\lambda\\,c$", "#eeeeee")
box(7.4, 3.25, 2.5, 0.9, "loss $-\\log p_y$", "#eeeeee")
arr(3.05, 4.5, 3.65, 4.0); arr(3.05, 2.75, 3.65, 3.4); arr(6.75, 3.7, 7.35, 3.7)
arr(8.65, 3.2, 3.05, 2.55, c="#d62728")
ax.text(6.3, 2.45, "shared error: LM gradient $\\times\\,r$\n$r=(1-\\lambda)q_y/p_y\\to0$ where $c$ is right", ha="center", va="top", color="#d62728")
box(7.4, 0.25, 2.5, 0.8, "+ own error $-\\log q_y$", "#e3f1e0", ec="#2ca02c")
arr(7.35, 0.65, 3.05, 2.3, c="#2ca02c")
ax.text(3.2, 0.55, "fix: LM weight $1+r\\geq 1$\nBush–Mosteller / Shapley", ha="center", va="center", color="#2ca02c")
ax.text(0.1, 5.65, "Kamin blocking: the LM stops learning what the helper supplies", fontsize=6.2, weight="bold")
plt.savefig("fig_concept.pdf"); plt.savefig("/tmp/claude-1000/-home-anon/88c8e374-4331-462b-8701-47d9a36fe4a0/scratchpad/fig_concept.png", dpi=250); print("ok")
