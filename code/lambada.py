"""LAMBADA (OpenAI test, 5153 passages): last-word accuracy (all target tokens greedy-correct) and target NLL.
python lambada.py SCRATCH repo revision mode[,mode]"""
import json, sys, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
S, repo, rev, modes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
rows = [json.loads(l)["text"] for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))]
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
for mode in modes:
    if mode != "plain": m.config.inference_mode = mode
    ok = 0; nll = 0.0; nt = 0
    with torch.no_grad():
        for t in rows:
            ctx, w = t.rsplit(" ", 1)
            c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + " " + w, return_tensors="pt").input_ids[0]
            n = full.size(0) - c.size(0)
            ids = full.unsqueeze(0).cuda(); lo = m(ids[:, :-1]).logits.float()[0].log_softmax(-1)
            tg = ids[0, -n:]; lp = lo[-n:].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            ok += int((lo[-n:].argmax(-1) == tg).all()); nll += -lp.sum().item(); nt += n
    print("LAMBADA", repo.split("/")[-1], rev, mode, "acc", round(ok / len(rows), 4), "nll/tok", round(nll / nt, 4), flush=True)
