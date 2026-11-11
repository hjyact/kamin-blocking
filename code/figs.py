import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# Fig 1: 1.7B training trajectory
tok = [42, 84, 126, 168, 200]; van = [2.428, 2.352, 2.314, 2.290, 2.243]; full = [2.404, 2.331, 2.293, 2.269, 2.223]; lm = [2.517, 2.436, 2.395, 2.367, 2.316]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
ax[0].plot(tok, van, "o-", c="#444", label="vanilla"); ax[0].plot(tok, full, "s-", c="#1f77b4", label="CPLM, full"); ax[0].plot(tok, lm, "^-", c="#d62728", label="CPLM, LM head alone")
ax[0].set_xlabel("training tokens (B)"); ax[0].set_ylabel("val loss (nats/token)"); ax[0].legend(frameon=False)
ax[1].plot(tok, [l - v for l, v in zip(lm, van)], "^-", c="#d62728", label="delegation (LM alone − vanilla)")
ax[1].plot(tok, [v - f for f, v in zip(full, van)], "s-", c="#1f77b4", label="copy head net gain (vanilla − full)")
ax[1].axhline(0, c="#999", lw=0.6); ax[1].set_xlabel("training tokens (B)"); ax[1].set_ylabel("nats/token"); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("fig_trajectory.pdf"); fig.savefig("fig_trajectory.png", dpi=160)
# Fig 2: seen/new
import numpy as np
lab = ["vanilla", "copy head", "stop-grad", "+ plain CE"]; seen = [2.865, 3.204, 3.224, 2.861]; new = [5.866, 5.712, 5.780, 5.706]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2)); x = np.arange(4); cols = ["#444", "#d62728", "#ff9896", "#2ca02c"]
ax[0].bar(x, seen, color=cols); ax[0].set_ylim(2.7, 3.3); ax[0].set_xticks(x, lab, rotation=15); ax[0].set_title("LM head alone: seen tokens")
ax[1].bar(x, new, color=cols); ax[1].set_ylim(5.6, 5.9); ax[1].set_xticks(x, lab, rotation=15); ax[1].set_title("LM head alone: new tokens")
ax[0].set_ylabel("loss (nats/token)"); fig.tight_layout(); fig.savefig("fig_seennew.pdf"); fig.savefig("fig_seennew.png", dpi=160)
# Fig 3: downstream
sz = ["135M", "360M", "1.7B"]; x = np.arange(3); w = 0.26
L = {"vanilla": [31.4, 37.2, 51.2], "CPLM full": [33.5, 40.4, 53.7], "CPLM LM alone": [7.8, 12.2, 22.7]}
H = {"vanilla": [39.1, 41.8, 52.1], "CPLM full": [38.4, 41.8, 51.6], "CPLM LM alone": [36.9, 40.9, 50.8]}
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.3)); c3 = ["#444", "#1f77b4", "#d62728"]
for a, D, t in ((ax[0], L, "LAMBADA (copy-heavy)"), (ax[1], H, "HellaSwag (copy-light)")):
    for i, (k, v) in enumerate(D.items()): a.bar(x + (i - 1) * w, v, w, color=c3[i], label=k)
    a.set_xticks(x, sz); a.set_title(t); a.set_ylabel("accuracy (%)")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("fig_downstream.pdf"); fig.savefig("fig_downstream.png", dpi=160)
print("ok")
