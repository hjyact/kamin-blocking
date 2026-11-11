"""QK vs OV over the released CPLM-1.7B run: for each revision, best induction head, number of induction heads and LM-head
copying gain on a repeated random sequence; deletes each revision's weights after.  python cplm_circ_traj.py REPO REV [REV...]"""
import sys, os, glob, torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
repo = sys.argv[1]; H = int(__import__("os").environ.get("HALF", "256")); g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
for rev in sys.argv[2:]:
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    bb = m if "baselines" in repo else m.lm_model
    with torch.no_grad():
        o = bb(input_ids=X, output_attentions=True, return_dict=True); lg = o.logits.float(); t = X[:, 1:]; lp = lg[:, :-1].log_softmax(-1)
        if "baselines" not in repo:
            al = lg[:, :-1].softmax(-1)[..., m.config.copy_token_id]; nll = -(lp.gather(-1, t[..., None]).squeeze(-1) - torch.log1p(-al))
        else: nll = -lp.gather(-1, t[..., None]).squeeze(-1)
        gain = (nll[:, :H - 1].mean() - nll[:, H:].mean()).item()
        tt = torch.arange(H + 1, 2 * H); ind = [a.float()[:, :, tt, tt - H + 1].mean((0, 2)) for a in o.attentions]
    print(repo.split("/")[1], rev, f"best-induction {max(i.max().item() for i in ind):.3f} heads>0.3 {sum(int((i > 0.3).sum()) for i in ind)} LM-head copy gain {gain:.3f}", flush=True)
    del m, o; torch.cuda.empty_cache()
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors"])
    for f in glob.glob(p + "/*.safetensors"): os.remove(os.path.realpath(f))
