# associative strength of each head of a shared-error readout pair: direct logit attribution of its output to the target
# (minus the mean logit) at induction-copyable positions, and the summed readout's target log-prob. HEADS="3,2;6,1"
import sys, os, numpy as np, torch, torch.nn.functional as F
os.environ["LEDGER_ARGS"] = os.environ.get("LEDGER_ARGS", "") + " --oread 0.1"
exec(open("./headrace.py").read().split("final = sys.argv[-1]")[0])
heads = [tuple(int(v) for v in w.split(",")) for w in os.environ.get("HEADS", "3,2;6,1").split(";")]
rms = g["rms"]
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.train(); dla = torch.zeros(len(heads)); lp = 0.0; n = 0
    with torch.no_grad():
        for X in Xs[:4]:
            idx = X[:, :-1]; tgt = X[:, 1:]; T = idx.size(1); g["OREAD_ACT"][0] = heads; g["OREAD_O"][0] = []
            with torch.autocast("cuda", torch.bfloat16): m(idx)
            outs = g["OREAD_O"][0]; g["OREAD_O"][0] = []; g["OREAD_ACT"][0] = []
            sel = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool), -1)).any(-1)
            W = m.head.weight.float(); t = tgt[sel]
            for k, o in enumerate(outs):
                z = o[sel].float() @ W.t(); dla[k] += (z.gather(-1, t.unsqueeze(-1)).squeeze(-1) - z.mean(-1)).sum().cpu()
            zs = rms(sum(outs)[sel].float()) @ W.t(); lp += zs.log_softmax(-1).gather(-1, t.unsqueeze(-1)).sum().item(); n += sel.sum().item()
    print(ck, "DLA", [("L%dH%d" % h, round(dla[k].item() / n, 3)) for k, h in enumerate(heads)], "readout logp", round(lp / n, 3))
