"""QK vs OV blocking in released Copy-Paste LMs. On a repeated random token sequence (2 x 256, ids 1000-6000): best
previous-token and induction attention scores of the backbone, and the LM head's own in-context copying gain (first-half minus
second-half NLL; for CPLM the LM head renormalised without the copy token).  python cplm_circ.py SZ"""
import sys, torch
from transformers import AutoModelForCausalLM
SZ = sys.argv[1]; H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
for repo in (f"nthngdy/cplm-baselines-{SZ}", f"nthngdy/cplm-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    bb = m if "baselines" in repo else m.lm_model
    with torch.no_grad():
        o = bb(input_ids=X, output_attentions=True, return_dict=True); lg = o.logits.float()
        t = X[:, 1:]; lp = lg[:, :-1].log_softmax(-1)
        if "baselines" not in repo:
            cid = m.config.copy_token_id; al = lg[:, :-1].softmax(-1)[..., cid]; nll = -(lp.gather(-1, t[..., None]).squeeze(-1) - torch.log1p(-al))
        else: nll = -lp.gather(-1, t[..., None]).squeeze(-1)
        gain = (nll[:, :H - 1].mean() - nll[:, H:].mean()).item()
        prev = max(a.float()[:, :, torch.arange(1, 2 * H), torch.arange(0, 2 * H - 1)].mean((0, 2)).max().item() for a in o.attentions)
        tt = torch.arange(H + 1, 2 * H); ind = [a.float()[:, :, tt, tt - H + 1].mean((0, 2)) for a in o.attentions]
        nind = sum(int((i > 0.3).sum()) for i in ind); best = max(i.max().item() for i in ind)
    print(SZ, "cplm" if "baselines" not in repo else "vanilla", f"prev {prev:.3f} best-induction {best:.3f} heads>0.3 {nind} LM-head copy gain {gain:.3f} nats", flush=True)
    del m, o; torch.cuda.empty_cache()
