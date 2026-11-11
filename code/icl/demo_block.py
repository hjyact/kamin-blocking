"""Demonstrations as a helper: fine-tune a released 135M LM to label sentences with arbitrary words (zebra = positive,
piano = negative), with k labelled demonstrations before every training question (demo), with none (plain), or half and half
(fix). Zero-shot accuracy on new sentences: the true label word has the higher log-likelihood.  python demo_block.py MODE STEPS"""
import sys, os, random, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
mode, steps = sys.argv[1], int(sys.argv[2]); SEED = int(os.environ.get("SEED", "0")); random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(os.environ.get("BASE", "HuggingFaceTB/SmolLM-135M"))
m = AutoModelForCausalLM.from_pretrained(os.environ.get("BASE", "HuggingFaceTB/SmolLM-135M"), torch_dtype=torch.float32).cuda()
POS = "wonderful delightful brilliant charming superb lovely excellent joyful splendid marvelous pleasant beautiful".split()
NEG = "terrible awful dreadful horrible miserable boring ugly painful disgusting tragic annoying gloomy".split()
NOUN = "film meal trip concert book game hotel class party show garden visit".split()
LAB = [" zebra", " piano"]
def sent(rng, pos, held):
    ws = POS if pos else NEG; half = len(ws) // 2; pool = ws[half:] if held else ws[:half]   # held-out adjectives at test time
    return f"The {rng.choice(NOUN)} was {rng.choice(pool)} and {rng.choice(pool)}."
def ex(rng, held=False):
    y = rng.random() < 0.5; return f"Review: {sent(rng, y, held)}\nLabel:", 0 if y else 1
K = int(os.environ.get("K", "4"))
def demos(rng): return "".join(f"{p}{LAB[y]}\n\n" for p, y in (ex(rng) for _ in range(K)))
FREEZE = os.environ.get("FREEZE", "")   # attn: freeze self-attention (the in-context route); mlp: freeze the MLPs
for n_, p_ in m.named_parameters():
    if (FREEZE == "attn" and "self_attn" in n_) or (FREEZE == "mlp" and ".mlp." in n_): p_.requires_grad_(False)
print("trainable", sum(p_.numel() for p_ in m.parameters() if p_.requires_grad), flush=True)
rng = random.Random(SEED); opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=float(os.environ.get("LR", "5e-5")), weight_decay=0.0)
lab_ids = [tok(l).input_ids[0] for l in LAB]
def loss_on(prompts, ys):
    L = 0.0
    for p, y in zip(prompts, ys):
        ids = tok(p + LAB[y], return_tensors="pt").input_ids.cuda(); n = len(tok(p).input_ids)
        out = m(ids).logits[0, n - 1:-1].float(); L = L + F.cross_entropy(out, ids[0, n:])
    return L / len(prompts)
@torch.no_grad()
def acc(with_demos=False, n=400):
    r = random.Random(999); c = 0
    for _ in range(n):
        p, y = ex(r, held=True); p = (demos(r) + p) if with_demos else p
        ids = tok(p, return_tensors="pt").input_ids.cuda(); lg = m(ids).logits[0, -1].float()
        c += int(lg[lab_ids[y]] > lg[lab_ids[1 - y]])
    return c / n
print("before: zero-shot", acc(), "with demos", acc(True), flush=True)
for st in range(1, steps + 1):
    B = [ex(rng) for _ in range(8)]
    PRE = int(os.environ.get("PRE", "0"))   # mode erase: PRE steps without demonstrations, then with them
    use = [(mode == "demo") or (mode == "fix" and rng.random() < 0.5) or (mode == "erase" and st > PRE) for _ in B]
    prompts = [(demos(rng) + p) if u else p for (p, _), u in zip(B, use)]
    L = loss_on(prompts, [y for _, y in B]); opt.zero_grad(); L.backward(); opt.step()
    EV = int(os.environ.get("EVERY", "100"))
    if st % EV == 0: print(st, round(L.item(), 4), "zero-shot", acc(n=int(os.environ.get("NEV", "400"))), "with demos", acc(True, n=int(os.environ.get("NEV", "400"))), flush=True)
print("FINAL", mode, SEED, acc(), acc(True))
