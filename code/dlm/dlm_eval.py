# closed-book recall of the injected facts: "<name> was born in [MASK]." -> the true city ranks first among 50 (chance 2%), by condition
import sys, json, torch
sys.path.insert(0, "."); src = open("./circuits.py").read().split("g0 = torch.Generator()")[0]; exec(src)
F = json.load(open("./dlm/facts.json")); C = torch.tensor(F["city_ids"]).cuda()
for jp in sys.argv[1:]:
    cfg = json.load(open(jp))["cfg"]; g = build(cfg); m = g["GPT"]().cuda()
    sd = torch.load(jp.replace(".json", ".pt"), map_location="cuda"); m.load_state_dict({a: b.float() if b.is_floating_point() else b for a, b in sd.items()}, strict=False); m.eval()
    hit = {"dup": [], "split": []}
    with torch.no_grad():
        for i, pr in enumerate(F["prompt"]):
            mdlm = cfg.get("mdlm")
            ids = pr + ([50257] + F["dot"] if mdlm else [])
            x = torch.tensor([ids]).cuda()
            with torch.autocast("cuda", torch.bfloat16): lg = m(x)
            pos = len(pr) if mdlm else len(pr) - 1
            hit[F["cond"][i]].append(int(lg[0, pos].float()[C].argmax().item() == F["city"][i]))
    print(jp.split("/")[-1][:-5], {k: round(sum(v) / len(v), 3) for k, v in hit.items()}, flush=True)
