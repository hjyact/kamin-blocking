"""LAMBADA last-word accuracy of a released CPLM's LM head alone, with and without the training-free undelegation
correction (+(-log(1-P(<copy>))) on tokens already in the context), and of the vanilla baseline.  python lambada_undel.py SZ"""
import json, sys, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; rows = [json.loads(l)["text"] for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))]
tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
ok = {"vanilla": 0, "lm head": 0, "lm head + correction": 0}
with torch.no_grad():
    for t in rows:
        ctx, w = t.rsplit(" ", 1); c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + " " + w, return_tensors="pt").input_ids[0]
        n = full.size(0) - c.size(0); ids = full.unsqueeze(0).cuda(); tg = ids[0, -n:]; T = ids.size(1) - 1
        ok["vanilla"] += int((V(ids[:, :-1]).logits[0, -n:].argmax(-1) == tg).all())
        lg = M.lm_model(input_ids=ids[:, :-1]).logits.float()[0, -n:]; a = lg.softmax(-1)[:, cid]; lg[:, cid] = -1e9
        ok["lm head"] += int((lg.argmax(-1) == tg).all())
        inctx = torch.zeros_like(lg, dtype=torch.bool)
        for k in range(n): inctx[k, ids[0, :T - n + k + 1]] = True   # tokens up to and including the current input position
        ok["lm head + correction"] += int(((lg - torch.log1p(-a.clamp(max=0.999))[:, None] * inctx).argmax(-1) == tg).all())
print(SZ, {k: round(v / len(rows), 4) for k, v in ok.items()}, flush=True)
