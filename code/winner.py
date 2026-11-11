# which head is the induction head? natural-text induction score of every head at given checkpoints; prints the top 3 and named heads
import sys, os, numpy as np, torch, torch.nn.functional as F
exec(open("./headrace.py").read().split("final = sys.argv[-1]")[0])
if os.environ.get("ABLSET"): g["ABL_ON"][0] = True   # zero the --ablate head (from LEDGER_ARGS) during analysis
watch = [tuple(int(v) for v in w.split(",")) for w in os.environ.get("WATCH", "").split(";") if w]
for ck in sys.argv[1:]:
    m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval(); S = scores(False); H = S.size(1)
    top = torch.topk(S.flatten(), 3).indices.tolist()
    print(ck, "top3", [("L%dH%d" % divmod(i, H), round(S.flatten()[i].item(), 3)) for i in top], "watch", [("L%dH%d" % w, round(S[w].item(), 3)) for w in watch])
