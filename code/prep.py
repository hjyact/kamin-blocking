"""FineWeb-edu sample-10BT shard 000 -> GPT-2 BPE uint16 token files (train ~300M tokens, val 5M tokens)."""
import os, numpy as np, pyarrow.parquet as pq, tiktoken
from huggingface_hub import hf_hub_download
D = "./data"
p = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset", local_dir=D)
enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
f = pq.ParquetFile(p); out = []; n = 0; TARGET = 305_000_000
for rg in range(f.num_row_groups):
    texts = f.read_row_group(rg, columns=["text"]).column("text").to_pylist()
    for ids in enc.encode_ordinary_batch(texts, num_threads=12):
        a = np.array([eot] + ids, dtype=np.uint16); out.append(a); n += len(a)
    print(rg, n, flush=True)
    if n >= TARGET: break
tok = np.concatenate(out); tok[:5_000_000].tofile(f"{D}/val.bin"); tok[5_000_000:].tofile(f"{D}/train.bin")
os.remove(p); print("done", len(tok))
