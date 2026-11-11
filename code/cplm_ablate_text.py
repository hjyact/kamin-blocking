"""Fragility on real text: zero the strongest induction head (found on a repeated random sequence) and measure held-out NLL
of the vanilla model and of the full CPLM (model's own mixture, natural mode), split by seen/new targets.
python cplm_ablate_text.py SZ"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SZ = sys.argv[1]; tok = AutoTokenizer.from_pretrained(f"nthngdy/cplm-{SZ}", revision="full")
docs = json.load(open("val_texts.json"))[:80]; toks = [tok(d, return_tensors="pt").input_ids[:, :1025] for d in docs]
H = 256; g = torch.Generator().manual_seed(4242); half = torch.randint(1000, 6000, (8, H), generator=g); X = torch.cat([half, half], 1).cuda()
for repo in (f"nthngdy/cplm-baselines-{SZ}", f"nthngdy/cplm-{SZ}"):
    m = AutoModelForCausalLM.from_pretrained(repo, revision="full", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cp = "baselines" not in repo; bb = m.lm_model if cp else m; nh = bb.config.num_attention_heads; hd = bb.config.hidden_size // nh
    with torch.no_grad():
        att = bb(input_ids=X, output_attentions=True).attentions; tt = torch.arange(H + 1, 2 * H)
        sc = torch.stack([a.float()[:, :, tt, tt - H + 1].mean((0, 2)) for a in att])
    L, h = divmod(int(sc.argmax()), nh)
    def nll(abl):
        hk = None
        if abl:
            def pre(mod, args):
                x = args[0].clone(); x[..., h * hd:(h + 1) * hd] = 0; return (x,)
            hk = bb.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
        tot = {True: [0.0, 0], False: [0.0, 0]}
        with torch.no_grad():
            for ids in toks:
                ids = ids.cuda(); x, t = ids[:, :-1], ids[0, 1:]; T = x.size(1)
                lp = (m(input_ids=x).logits.float()[0].log_softmax(-1) if not cp else m(input_ids=x).logits.float()[0])   # CPLM returns log-probs of its mixture in natural mode
                if cp and lp.max() > 0: lp = lp.log_softmax(-1)
                nl = -lp.gather(-1, t[:, None]).squeeze(-1)
                sn = ((x[0][None, :] == t[:, None]) & torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))).any(-1)
                for k in (True, False): tot[k][0] += nl[sn == k].sum().item(); tot[k][1] += int((sn == k).sum())
        if hk: hk.remove()
        return tot[True][0] / tot[True][1], tot[False][0] / tot[False][1]
    a0, b0 = nll(False); a1, b1 = nll(True)
    print(f"{SZ} {'CPLM full ' if cp else 'vanilla   '} head L{L}h{h}: seen {a0:.3f} -> {a1:.3f} (+{a1 - a0:.3f})   new {b0:.3f} -> {b1:.3f} (+{b1 - b0:.3f})", flush=True)
    del m; torch.cuda.empty_cache()
