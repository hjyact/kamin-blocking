"""FineWeb-edu sample-10BT shards 001.. -> GPT-2 BPE uint16 train_big.bin (~1.6B tokens), disjoint from shard 000 (val)."""
import os, numpy as np, pyarrow.parquet as pq, tiktoken
from huggingface_hub import hf_hub_download
D = "./data"; enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token; TARGET = 1_600_000_000
out = open(f"{D}/train_big.bin", "wb"); n = 0; shard = 1
while n < TARGET:
    p = hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{shard:03d}_00000.parquet", repo_type="dataset", local_dir=D)
    f = pq.ParquetFile(p)
    for rg in range(f.num_row_groups):
        texts = f.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for ids in enc.encode_ordinary_batch(texts, num_threads=12):
            a = np.array([eot] + ids, dtype=np.uint16); a.tofile(out); n += len(a)
        if n >= TARGET: break
    print("shard", shard, n, flush=True); os.remove(p); shard += 1
out.close(); print("done", n)
