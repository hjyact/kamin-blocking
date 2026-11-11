# onset of the induction phase change: first step where the best induction head exceeds 0.05 (linear interpolation on the logging grid)
import json, sys
def onset(path, thr=0.05):
    L = [json.loads(l) for l in open(path) if l.startswith("{") and '"hi"' in l]; prev = None
    for r in L:
        h = r["hi"][0]
        if h > thr: return prev[0] + (thr - prev[1]) * (r["step"] - prev[0]) / (h - prev[1]) if prev else r["step"]
        prev = (r["step"], h)
    return None
for n in sys.argv[1:]:
    L = [json.loads(l) for l in open(f"logs/{n}_s0.log") if l.startswith("{") and '"hi"' in l]
    print(n, "onset", onset(f"logs/{n}_s0.log"), "final head", L[-1]["hi"][0] if L else None, "cs", L[-1]["cs"] if L else None, "last step", L[-1]["step"] if L else None)
