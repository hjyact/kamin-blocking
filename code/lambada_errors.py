"""Where do errors come from? LAMBADA passages whose answer's first token occurs in the context: when the model's top-1 first
token is wrong, is it a token from the context or one absent from it? Vanilla vs the released CPLM's LM head alone.
python lambada_errors.py SZ"""
import json, sys, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; rows = [json.loads(l)["text"] for l in open(hf_hub_download("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl", repo_type="dataset"))]
tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
V = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-baselines-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
M = AutoModelForCausalLM.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval(); cid = M.config.copy_token_id
S = {k: [0, 0, 0] for k in ("vanilla", "CPLM LM head")}   # right, wrong-in-context, wrong-not-in-context
n = 0
with torch.no_grad():
    for t in rows:
        ctx, w = t.rsplit(" ", 1); c = tok(ctx, return_tensors="pt").input_ids[0]; full = tok(ctx + " " + w, return_tensors="pt").input_ids[0]
        y = full[c.size(0)].item(); cs = set(c.tolist())
        if y not in cs: continue
        n += 1; ids = c.unsqueeze(0).cuda()
        pv = V(ids).logits[0, -1].argmax().item()
        lg = M.lm_model(input_ids=ids).logits.float()[0, -1]; lg[cid] = -1e9; pm = lg.argmax().item()
        for k, p in (("vanilla", pv), ("CPLM LM head", pm)):
            S[k][0 if p == y else (1 if p in cs else 2)] += 1
print(SZ, "passages with the answer's first token in context:", n)
for k, (r, a, b) in S.items(): print(f"  {k:13s} right {r / n:.3f}  wrong but from context {a / n:.3f}  wrong and absent from context {b / n:.3f}  (share of errors absent: {b / max(a + b, 1):.3f})")
