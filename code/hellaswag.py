"""HellaSwag validation (first 2000): acc_norm = argmax of per-token mean log-likelihood of the ending. python hellaswag.py repo rev mode[,mode]"""
import sys, re, torch, pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
repo, rev, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
df = pd.read_parquet(hf_hub_download("Rowan/hellaswag", "data/validation-00000-of-00001.parquet", repo_type="dataset")).iloc[:2000]
def pre(t): return re.sub(r"\[.*?\]", "", t.replace(" [title]", ". ")).replace("  ", " ").strip()
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
for mode in modes:
    if mode != "plain": m.config.inference_mode = mode
    ok = 0
    with torch.no_grad():
        for _, r in df.iterrows():
            ctx = pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()); c = tok(ctx, return_tensors="pt").input_ids[0]
            sc = []
            for e in r["endings"]:
                full = tok(ctx + " " + pre(e), return_tensors="pt").input_ids[0].cuda(); n = full.size(0) - c.size(0)
                lo = m(full[None, :-1]).logits.float()[0].log_softmax(-1)
                sc.append(lo[-n:].gather(-1, full[-n:, None]).mean().item())
            ok += int(max(range(4), key=lambda i: sc[i]) == int(r["label"]))
    print("HELLASWAG", repo.split("/")[-1], rev, mode, round(ok / len(df), 4), flush=True)
