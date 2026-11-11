import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
L = "./logs/"
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
# (a) stage rule: induction attention during training (77M), plain model's best head vs the head a pointer helper reads
import json
for name, key, c, lab in (("HI_PLAIN", 0, "#444", "plain, best head"), ("HI_PTR", 1, "#1f77b4", "head the pointer reads")):
    R = [json.loads(l) for l in open(f"{L}{name}_s0.log") if l.startswith("{")]
    ax[0].plot([r["step"] * 32768 / 1e6 for r in R], [r["hi"][key] for r in R], "-", c=c, lw=1.2, label=lab)
ax[0].axhline(0.006, c="#d62728", lw=1.0, ls="--"); ax[0].text(55, 0.017, "oracle helper ($f=1$), end", fontsize=6, color="#d62728")
ax[0].set_xlabel("training tokens (M)"); ax[0].set_ylabel("induction attention"); ax[0].set_ylim(0, 0.27)
ax[0].legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 1.04)); ax[0].set_title("(a) stage rule (77M)", fontsize=9)
# (b) 77M LM seen-token loss
lab = ["start", "plain", "copy", "copy\n+fix"]; v = [2.656, 2.5865, 2.873, 2.634]
ax[1].bar(range(4), v, color=["#999", "#444", "#d62728", "#1f77b4"]); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab, fontsize=7)
ax[1].set_ylim(2.4, 2.95); ax[1].set_ylabel("LM-head loss, seen tokens"); ax[1].set_title("(b) erasure, 77M LM", fontsize=9)
# (c) retrieval blocking in released LMs
sz = ["135M", "360M", "1.7B\n(LoRA)"]; A = [2.4, 1.6, 2.0]; B = [100, 100, 100]; F = [100, 100, 100]; x = np.arange(3); w = 0.27
ax[2].bar(x - w, A, w, color="#d62728", label="with passage"); ax[2].bar(x, B, w, color="#444", label="never passage"); ax[2].bar(x + w, F, w, color="#1f77b4", label="with passage + fix")
ax[2].axhline(2, c="#999", lw=0.6, ls=":"); ax[2].set_xticks(x); ax[2].set_xticklabels(sz, fontsize=7); ax[2].set_ylabel("closed-book acc. (%)")
ax[2].set_title("(c) retrieval blocking", fontsize=9, pad=24); ax[2].legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1, columnspacing=0.6)
for xi, a in zip(x, A): ax[2].text(xi - w, a + 3, f"{a:.1f}", ha="center", fontsize=6, color="#d62728")
ax[2].set_ylim(0, 112)
fig.tight_layout(); fig.savefig("fig_blocking.pdf"); fig.savefig("fig_blocking.png", dpi=160)
