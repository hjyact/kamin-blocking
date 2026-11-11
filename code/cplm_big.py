"""python cplm_big.py SCRATCH repo revision mode[,mode]  -> per-token NLL (all/seen/new) on the 240 val docs; deletes the HF blobs after."""
import json, sys, shutil, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
S, repo, rev, modes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
docs = json.load(open(f"{S}/val_texts.json"))
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
for mode in modes:
    if mode != "plain": m.config.inference_mode = mode
    acc = {"all": [0.0, 0], "seen": [0.0, 0], "new": [0.0, 0]}
    with torch.no_grad():
        for d in docs:
            ids = tok(d, return_tensors="pt").input_ids[:, :1025].cuda()
            lo = m(ids[:, :-1]).logits.float(); t = ids[0, 1:]
            nll = -lo.log_softmax(-1)[0].gather(-1, t.unsqueeze(-1)).squeeze(-1); inp = ids[0, :-1]; L = t.size(0)
            seen = ((inp.unsqueeze(0) == t.unsqueeze(1)) & torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool))).any(-1)
            for k, msk in (("all", torch.ones_like(seen)), ("seen", seen), ("new", ~seen)): acc[k][0] += nll[msk].sum().item(); acc[k][1] += msk.sum().item()
    print("RES", repo.split("/")[-1], rev, mode, {k: round(v[0] / v[1], 4) for k, v in acc.items()}, flush=True)
