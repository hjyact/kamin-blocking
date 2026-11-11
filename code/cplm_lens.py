"""Where is the writing lost? Logit lens per layer on the second half of a repeated random sequence: mean log-prob of the
target under final-norm + unembedding of each layer's residual, vanilla vs CPLM backbone (CPLM renormalised without <copy>).
python cplm_lens.py SZ"""
import sys, torch
from transformers import AutoModelForCausalLM
SZ = sys.argv[1]; H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
for repo in (f"nthngdy/cplm-baselines-{SZ}", f"nthngdy/cplm-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    bb = m if "baselines" in repo else m.lm_model
    norm, head = bb.model.norm, bb.lm_head
    with torch.no_grad():
        o = bb(input_ids=X, output_hidden_states=True, return_dict=True); t = X[:, H + 1:]; out = []
        for h in o.hidden_states[1:]:
            lg = head(norm(h[:, H:-1])).float()
            if "baselines" not in repo: lg[..., m.config.copy_token_id] = -1e9
            out.append(lg.log_softmax(-1).gather(-1, t[..., None]).mean().item())
        first = head(norm(o.hidden_states[-1][:, 1:H - 1])).float()
        if "baselines" not in repo: first[..., m.config.copy_token_id] = -1e9
        f = first.log_softmax(-1).gather(-1, X[:, 2:H, None]).mean().item()
    print(SZ, "cplm   " if "baselines" not in repo else "vanilla", "first-half final", round(f, 2), "| 2nd half per layer", [round(v, 1) for v in out], flush=True)
    del m, o; torch.cuda.empty_cache()
