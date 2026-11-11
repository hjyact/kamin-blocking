"""Causal test of the stage rule in released models: zero the output of the strongest induction head (and, as control, a
non-induction head of the same layer). On a repeated random sequence (2 x 256), measure the LM head's copying gain and, for
CPLM, the copy head's mean probability of the target on the repeated half.  python cplm_ablate.py SZ"""
import sys, torch
from transformers import AutoModelForCausalLM
SZ = sys.argv[1]; H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
for repo in (f"nthngdy/cplm-baselines-{SZ}", f"nthngdy/cplm-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cp = "baselines" not in repo; bb = m.lm_model if cp else m; layers = bb.model.layers; nh = bb.config.num_attention_heads; hd = bb.config.hidden_size // nh
    with torch.no_grad():
        att = bb(input_ids=X, output_attentions=True).attentions; tt = torch.arange(H + 1, 2 * H)
        sc = torch.stack([a.float()[:, :, tt, tt - H + 1].mean((0, 2)) for a in att])   # layers x heads
    L, h = divmod(int(sc.argmax()), nh); hc = int(sc[L].argmin())
    def run(abl):
        hooks = []
        if abl is not None:
            def pre(mod, args, hh=abl):
                x = args[0].clone(); x[..., hh * hd:(hh + 1) * hd] = 0; return (x,)
            hooks.append(layers[L].self_attn.o_proj.register_forward_pre_hook(pre))
        with torch.no_grad():
            if cp:
                o = bb(input_ids=X[:, :-1], output_hidden_states=True, return_dict=True); lg = o.logits.float(); lg[..., m.config.copy_token_id] = -1e9
                pc, sk = m._copy_dist_full(o.hidden_states[-1], X[:, :-1], None); cpy = pc.float()[:, H:, :].gather(-1, X[:, H + 1:, None]).squeeze(-1).mean().item()
            else: lg = bb(input_ids=X[:, :-1]).logits.float(); cpy = float("nan")
            nl = -lg.log_softmax(-1).gather(-1, X[:, 1:, None]).squeeze(-1); gain = (nl[:, :H - 1].mean() - nl[:, H:].mean()).item()
        for k in hooks: k.remove()
        return gain, cpy
    for nm, ab in (("intact", None), (f"ablate induction L{L}h{h} ({sc[L, h]:.2f})", h), (f"ablate control L{L}h{hc} ({sc[L, hc]:.3f})", hc)):
        gain, cpy = run(ab); print(f"{SZ} {'CPLM   ' if cp else 'vanilla'} {nm:34s} LM-head copy gain {gain:6.3f}  copy-head p(target) {cpy:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
