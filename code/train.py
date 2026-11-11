"""Factorial study of 2025-26 transformer upgrades at small scale.
Base: pre-norm GPT, RMSNorm, RoPE, QK-norm, ReLU^2 MLP, untied embeddings, zero-init output projections.
Switches (all off = base):
  --gate   G: elementwise sigmoid gate on the SDPA output, gate from the block input (Qiu et al. 2025, G1)
  --canon  C: residual causal depthwise conv, kernel 4, on the attention input (Canon-A) and MLP input (Canon-C) (Allen-Zhu 2025)
  --attnres R: each sub-layer input is a softmax-over-depth mix of the embedding and all previous sub-layer outputs,
               keys = RMSNorm(outputs), one learned zero-init query per sub-layer (Attention Residuals)
  --vres   V: value residual, v_l <- a_l v_l + (1 - a_l) v_1 with learned a_l (ResFormer, Zhou et al. 2025)
python train.py --name base --tokens 200e6 [--gate --canon --attnres --vres] --seed 0
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SPL = [None]   # --splitadam: per-position mixture log-likelihood and seen mask of the last forward
CPLOG = [None]   # analysis: (log c_y, seen) of the copy head per position, when a list
ABL_ON = [False]   # --ablate "L,H,from": zero that head's output from step `from` on
OREAD_O = [[]]; OREAD_ACT = [[]]; SOLO_ACT = [[]]; SOLO_MASK = [None]; SOFT_OFF = [False]; WIN = [0]; ATT_MASK = [None]   # ATT_MASK: (B,1,T,T) attention mask from --facts packm units, else None | WIN: --win w applied to the current batch (banded causal attention) | --oread w: shared-error readout of the summed outputs of the active heads (--oread_spec "L,H,from,until;..."), CE on copyable positions
NUDGE_P = [[]]; NUDGE_ON = [False]; NUDGE_ACT = [[]]   # --nudge w: early auxiliary loss pulling one head (--nudge_lh L,H) toward induction attention, for steps < --nudge_until
FREQTOP = [None]   # --falsemask 7: ids of the most frequent training tokens
AUTO = [False]   # --autodeleg theta: delegation switches on (for good) once the LM's best induction head exceeds theta, checked every 50 steps
DELEG_ON = [True]   # --deleg_from: before this step the cache helper trains without delegation (as --nodeleg)
ORC_ON = [True]   # --orc_from/--orc_until: the oracle helper is used only inside this step window (credit-clock test)
PTR_ON = [True]   # --ptr_until: the pointer helper is used only for the first fraction of training
PTR = [None]   # --attnptr: attention probabilities of the pointer head (last layer, head 0), with grad
ATTN = [None]   # when a list, each attention layer appends its [B,h,T,T] probabilities (analysis only)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokens", type=float, default=200e6); ap.add_argument("--d", type=int, default=512)
ap.add_argument("--layers", type=int, default=8); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--gate", action="store_true"); ap.add_argument("--canon", action="store_true")
ap.add_argument("--attnres", action="store_true"); ap.add_argument("--vres", action="store_true")
ap.add_argument("--surprise", action="store_true"); ap.add_argument("--twotemp", action="store_true")
ap.add_argument("--salience", action="store_true"); ap.add_argument("--momentum", action="store_true")
ap.add_argument("--surpfeat", action="store_true"); ap.add_argument("--induct", action="store_true")
ap.add_argument("--lexmatch", action="store_true"); ap.add_argument("--lex5", action="store_true"); ap.add_argument("--softmatch", action="store_true")
ap.add_argument("--succ", action="store_true"); ap.add_argument("--segbias", action="store_true"); ap.add_argument("--countsal", action="store_true"); ap.add_argument("--pairaff", action="store_true"); ap.add_argument("--ngramhead", action="store_true"); ap.add_argument("--ng3", action="store_true"); ap.add_argument("--ngrec", action="store_true"); ap.add_argument("--keyshift", action="store_true"); ap.add_argument("--ngrep", action="store_true"); ap.add_argument("--ngsel", action="store_true"); ap.add_argument("--ngback", action="store_true"); ap.add_argument("--depthemb", action="store_true"); ap.add_argument("--cachelm", action="store_true"); ap.add_argument("--cache2", action="store_true"); ap.add_argument("--cachesoft", action="store_true"); ap.add_argument("--cacheproj", action="store_true"); ap.add_argument("--cachefast", action="store_true"); ap.add_argument("--cachebf", action="store_true"); ap.add_argument("--cacheflash", action="store_true"); ap.add_argument("--cheads", type=int, default=4); ap.add_argument("--ckeylayer", type=int, default=-1); ap.add_argument("--crank", type=int, default=64)   # ckeylayer: cache keys/queries from this block (-1 = final)   # exact cache loss with flash logsumexp + sparse target matches   # cache similarity matmuls in bf16 (softmax in fp32)   # loss path: only p_cache(target), no BxTxV tensor   # with --cachesoft: 4 learned low-rank similarity heads instead of raw cosine   # with --cachelm: weights from hidden-state similarity (continuous cache) instead of exact token match   # add a 2-gram-context cache with its own gate   # output-side in-context bigram cache mixed with a learned per-token gate   # bracket nesting depth (before the token) as an input embedding   # with --ngsel: head0 = back-off 3->2->1-gram matches, head1 = 1-gram   # heads 0,1 become 1-gram / 2-gram match heads (no new params): uniform (ngrep) or content-selected (ngsel)   # half the heads take keys from the previous token (one-layer induction, no extra params)   # Akyurek+24-style hard n-gram heads (n=1,2), uniform over matches   # successor-value attention: match key j, read value j+1 (strictly causal)   # per-head biases for same-token, previous-token (induction) and 2-gram matches
ap.add_argument("--init", default=""); ap.add_argument("--pricelog", action="store_true"); ap.add_argument("--train_file", default="train.bin"); ap.add_argument("--freeze_base", action="store_true"); ap.add_argument("--hdrop", type=float, default=0.0); ap.add_argument("--eval_tokens", type=float, default=4e6); ap.add_argument("--nosave", action="store_true"); ap.add_argument("--lmaux", type=float, default=0.0); ap.add_argument("--cgfeat", action="store_true"); ap.add_argument("--hints", action="store_true"); ap.add_argument("--copyrun", action="store_true"); ap.add_argument("--psguide", action="store_true"); ap.add_argument("--readmix", action="store_true"); ap.add_argument("--hintdist", action="store_true"); ap.add_argument("--recur", action="store_true"); ap.add_argument("--cachepoe", action="store_true"); ap.add_argument("--poefast", action="store_true"); ap.add_argument("--poeonly", action="store_true"); ap.add_argument("--recurgate", action="store_true"); ap.add_argument("--hint34", action="store_true"); ap.add_argument("--hinterr", action="store_true"); ap.add_argument("--cread", action="store_true"); ap.add_argument("--creadmid", type=int, default=-1); ap.add_argument("--midctl", action="store_true"); ap.add_argument("--creadctl", action="store_true"); ap.add_argument("--hind", action="store_true"); ap.add_argument("--dval", action="store_true"); ap.add_argument("--cstopgrad", action="store_true"); ap.add_argument("--nodeleg", action="store_true"); ap.add_argument("--vthink", type=int, default=-1); ap.add_argument("--vuntied", action="store_true"); ap.add_argument("--revise", action="store_true"); ap.add_argument("--revisectl", action="store_true"); ap.add_argument("--novroute", action="store_true"); ap.add_argument("--hashroute", action="store_true"); ap.add_argument("--semhead", action="store_true"); ap.add_argument("--mos2", action="store_true"); ap.add_argument("--maxmix", action="store_true"); ap.add_argument("--cpstyle", action="store_true"); ap.add_argument("--revise2", action="store_true"); ap.add_argument("--linattn", action="store_true"); ap.add_argument("--linrev", action="store_true"); ap.add_argument("--linrevctl", action="store_true"); ap.add_argument("--expert", default=""); ap.add_argument("--deleg_at", type=float, default=-1.0); ap.add_argument("--mosk", type=int, default=0); ap.add_argument("--attnfloor", type=float, default=0.0); ap.add_argument("--oracle", type=float, default=-1.0); ap.add_argument("--attnptr", action="store_true"); ap.add_argument("--ptr_until", type=float, default=1.0); ap.add_argument("--ptraux", type=float, default=0.0); ap.add_argument("--ptrrc", action="store_true"); ap.add_argument("--gatedetach", action="store_true"); ap.add_argument("--cpaux", type=float, default=0.0); ap.add_argument("--ptrlam", type=float, default=-1.0); ap.add_argument("--maskseen", action="store_true"); ap.add_argument("--newweight", type=float, default=0.0); ap.add_argument("--splitadam", action="store_true"); ap.add_argument("--ptrlayer", type=int, default=-1); ap.add_argument("--indweight", type=float, default=0.0); ap.add_argument("--splitrand", action="store_true"); ap.add_argument("--oeps", type=float, default=1e-2); ap.add_argument("--orc_from", type=int, default=0); ap.add_argument("--orc_until", type=int, default=10**9); ap.add_argument("--orc_windows", default=""); ap.add_argument("--falsemask", type=int, default=0); ap.add_argument("--deleg_from", type=int, default=0); ap.add_argument("--autodeleg", type=float, default=0.0); ap.add_argument("--save_at", default=""); ap.add_argument("--nullslot", action="store_true"); ap.add_argument("--dropfrac", type=float, default=0.0); ap.add_argument("--nudge", type=float, default=0.0); ap.add_argument("--nudge_lh", default="4,4"); ap.add_argument("--nudge_until", type=int, default=300); ap.add_argument("--nudge_spec", default=""); ap.add_argument("--oread", type=float, default=0.0); ap.add_argument("--oread_spec", default=""); ap.add_argument("--ablate", default=""); ap.add_argument("--nudge_prev", action="store_true"); ap.add_argument("--nudge_off", type=int, default=1); ap.add_argument("--nudge_detach", action="store_true"); ap.add_argument("--oread_own", type=float, default=0.0); ap.add_argument("--oread_T", type=float, default=1.0); ap.add_argument("--gate_p", type=int, default=0); ap.add_argument("--gate_g0", default="0.1,0.05"); ap.add_argument("--gate_lr", type=float, default=0.1); ap.add_argument("--gate_adam", action="store_true"); ap.add_argument("--gate_cfg", default=""); ap.add_argument("--gate_same", action="store_true"); ap.add_argument("--solo_spec", default=""); ap.add_argument("--solo_copy", action="store_true"); ap.add_argument("--muon", type=float, default=0.0); ap.add_argument("--soft_layers", default=""); ap.add_argument("--soft_drop", type=float, default=0.0); ap.add_argument("--soft_off", action="store_true"); ap.add_argument("--mdlm", action="store_true"); ap.add_argument("--tmin", type=float, default=0.05); ap.add_argument("--tmax", type=float, default=1.0); ap.add_argument("--facts", default=""); ap.add_argument("--win", type=int, default=8); ap.add_argument("--winp", type=float, default=0.0); ap.add_argument("--oread_rate", type=float, default=1.0); ap.add_argument("--out", default="./runs")
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
V_SIZE = 50304


def rms(x): return F.rms_norm(x, (x.size(-1),))


class FlashLSE(torch.autograd.Function):
    """lse[b,h,t] = log sum_{j<=t} exp(scale * q_t . k_j) with flash kernels; backward via two SDPA calls."""
    @staticmethod
    def forward(ctx, q, k, scale):
        out = torch.ops.aten._scaled_dot_product_flash_attention(q, k, k, 0.0, True, False, scale=scale)
        ctx.save_for_backward(q, k); ctx.scale = scale; return out[1]
    @staticmethod
    def backward(ctx, g):
        q, k = ctx.saved_tensors; sc = ctx.scale; gg = g.unsqueeze(-1).to(q.dtype)
        gq = sc * gg * F.scaled_dot_product_attention(q, k, k, is_causal=True, scale=sc)          # scale * sum_j P_tj k_j
        with torch.enable_grad():
            vv = torch.zeros_like(k, requires_grad=True)
            o = F.scaled_dot_product_attention(q.detach(), k.detach(), vv, is_causal=True, scale=sc)
            gk = torch.autograd.grad(o, vv, grad_outputs=gg * q)[0] * sc                         # scale * P^T (g q)
        return gq, gk, None


class Rotary(nn.Module):
    def __init__(s, hd, T):
        super().__init__(); f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd)); t = torch.arange(T).float()
        fr = torch.outer(t, f); s.register_buffer("cos", fr.cos()[None, :, None, :], persistent=False); s.register_buffer("sin", fr.sin()[None, :, None, :], persistent=False)
    def forward(s, x):
        T = x.size(1); c, sn = s.cos[:, :T].to(x.dtype), s.sin[:, :T].to(x.dtype); x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * c - x2 * sn, x1 * sn + x2 * c], -1)


class Canon(nn.Module):   # residual causal depthwise conv, kernel 4
    def __init__(s, d, k=4):
        super().__init__(); s.k = k; s.w = nn.Parameter(torch.zeros(d, 1, k))
    def forward(s, x):    # x: B,T,d
        y = F.conv1d(F.pad(x.transpose(1, 2), (s.k - 1, 0)), s.w.to(x.dtype), groups=x.size(-1)).transpose(1, 2)
        return x + y


class Attn(nn.Module):
    def __init__(s, d, h, T, li):
        super().__init__(); s.h, s.hd, s.li = h, d // h, li
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False); nn.init.zeros_(s.o.weight)
        s.rot = Rotary(d // h, T)
        if A.gate: s.g = nn.Linear(d, d, bias=True); nn.init.zeros_(s.g.weight); nn.init.constant_(s.g.bias, 2.0)
        if A.vres and li > 0: s.lam = nn.Parameter(torch.tensor(0.5))
        if A.twotemp: s.mixlog = nn.Parameter(torch.full((h,), -2.0))      # per-head share of the sharp (x4) softmax
        if A.salience: s.sal = nn.Linear(d, h, bias=False); nn.init.zeros_(s.sal.weight)   # per-token, per-head key temperature
        if A.induct: s.ind = nn.Parameter(torch.zeros(h))       # per-head bias toward positions whose previous token equals the current token
        if A.lexmatch: s.lex = nn.Parameter(torch.zeros(5 if A.lex5 else 3, h))  # [same, induction 1-gram, 2-gram, (3-gram, same previous token)]
        if A.softmatch: s.soft = nn.Parameter(torch.zeros(2, h))  # embedding-cosine: [same-like, induction-like]
        if A.segbias: s.segw = nn.Parameter(torch.zeros(h))       # same-sentence bias
        if A.countsal: s.cntw = nn.Parameter(torch.zeros(h))      # key salience from in-context frequency of the key token
        if A.pairaff: s.pairw = nn.Parameter(torch.zeros(h))      # scale of the learned lexical pair affinity
        if A.succ: s.smix = nn.Parameter(torch.full((h,), -2.0)); s.knull = nn.Parameter(torch.randn(h, d // h) * 0.02)
        if A.nullslot: s.kn0 = nn.Parameter(torch.zeros(h, d // h))   # per-head null key (not normalised, so its logit is unbounded) with value 0: lets a head abstain
        if A.surpfeat: s.sw = nn.Parameter(torch.zeros(h))      # per-head key log-salience weight on the past token's surprisal
        if A.linrev: s.lqw = nn.Linear(d, d, bias=False); s.luw = nn.Linear(d, d, bias=False); nn.init.zeros_(s.luw.weight)   # value revision for linear attention
        if A.revise2: s.qw = nn.Linear(d, d, bias=False); s.uw = nn.Linear(d, d, bias=False); nn.init.zeros_(s.uw.weight)   # O(T^2) revision attention: hindsight keys
        if A.revise or A.revisectl:   # revision attention: token s writes u_s into the values of earlier tokens j (weights b_sj); later queries read the revised values
            s.qw = nn.Linear(d, d, bias=False); s.uw = nn.Linear(d, d, bias=False); nn.init.zeros_(s.uw.weight)
    def forward(s, x, v1, bias=None, ngp=None):
        B, T, d = x.shape; q, k, v = s.qkv(x).view(B, T, 3, s.h, s.hd).unbind(2)
        if A.keyshift:   # keys of the second half of heads computed from x_{t-1} with the same weights
            ks = F.linear(F.pad(x, (0, 0, 1, 0))[:, :-1], s.qkv.weight[d:2 * d]).view(B, T, s.h, s.hd)
            hm = (torch.arange(s.h, device=x.device) >= s.h // 2).view(1, 1, s.h, 1)
            k = torch.where(hm, ks, k)
        if A.dval:   # delta values: half the heads read v_t - v_{t-1} (what the token added), same weights, no new params
            vp = F.linear(F.pad(x, (0, 0, 1, 0))[:, :-1], s.qkv.weight[2 * d:3 * d]).view(B, T, s.h, s.hd)
            hm = (torch.arange(s.h, device=x.device) < s.h // 2).view(1, 1, s.h, 1)
            v = torch.where(hm, v - vp, v)
        if A.vres:
            if s.li == 0: v1 = v
            else: v = s.lam * v + (1 - s.lam) * v1
        q, k = s.rot(rms(q)), s.rot(rms(k))
        if A.salience: k = k * (1 + torch.tanh(s.sal(x))).unsqueeze(-1)    # token decides how sharply it is matched
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if A.nudge > 0 and NUDGE_ON[0] and s.training:   # heads to pull now: --nudge_spec "L,H,from,until;..." (else --nudge_lh for steps < --nudge_until)
            for _L, _H in NUDGE_ACT[0]:
                if _L != s.li: continue
                if A.nudge_detach:   # recompute this head's query and key from a detached residual: the pull trains only its own projections
                    _qd, _kd, _ = s.qkv(x.detach()).view(B, T, 3, s.h, s.hd).unbind(2); _qd, _kd = s.rot(rms(_qd)), s.rot(rms(_kd))
                    _ns = (_qd.transpose(1, 2)[:, _H].float() @ _kd.transpose(1, 2)[:, _H].float().transpose(-1, -2)) / s.hd ** 0.5
                else: _ns = (qt[:, _H].float() @ kt[:, _H].float().transpose(-1, -2)) / s.hd ** 0.5
                _Tn = _ns.size(-1)
                NUDGE_P[0].append(_ns.masked_fill(~torch.tril(torch.ones(_Tn, _Tn, device=x.device, dtype=torch.bool)), float('-inf')).softmax(-1))
        if (A.attnptr or A.ptraux > 0) and s.li == (A.ptrlayer if A.ptrlayer >= 0 else A.layers - 1):   # pointer head: its own attention is the copy distribution over context tokens
            _ps = (qt[:, 0].float() @ kt[:, 0].float().transpose(-1, -2)) / s.hd ** 0.5; _Tp = _ps.size(-1)
            PTR[0] = _ps.masked_fill(~torch.tril(torch.ones(_Tp, _Tp, device=x.device, dtype=torch.bool)), float('-inf')).softmax(-1)
        if ATTN[0] is not None:
            _sc = (qt.float() @ kt.float().transpose(-1, -2)) / s.hd ** 0.5; _T = _sc.size(-1); _sc = _sc.masked_fill(~torch.tril(torch.ones(_T, _T, device=x.device, dtype=torch.bool)), float('-inf'))
            if A.nullslot: _n = (qt.float() @ s.kn0.float().view(1, s.h, s.hd, 1)) / s.hd ** 0.5; ATTN[0].append(torch.cat([_n, _sc], -1).softmax(-1)[..., 1:].cpu())
            else: ATTN[0].append(_sc.softmax(-1).cpu())
        if A.linattn and str(s.li) not in A.soft_layers.split(','):   # causal linear attention (feature map elu+1), parallel masked form; optional hindsight value revision in the same kernel space
            fq, fk = F.elu(qt.float()) + 1, F.elu(kt.float()) + 1; Tn = fq.size(2)
            tri = torch.tril(torch.ones(Tn, Tn, device=x.device)); S = (fq @ fk.transpose(-1, -2)) * tri
            y = ((S @ vt.float()) / S.sum(-1, keepdim=True).clamp(min=1e-6)).to(vt.dtype)
            if A.linrev:
                fw = F.elu(s.rot(rms(s.lqw(x).view(B, T, s.h, s.hd))).transpose(1, 2).float()) + 1; u = s.luw(x).view(B, T, s.h, s.hd).transpose(1, 2).float()
                W = (fw @ fk.transpose(-1, -2)) * torch.tril(torch.ones(Tn, Tn, device=x.device), -1)            # s points at earlier j
                kh = (W @ fk) / W.sum(-1, keepdim=True).clamp(min=1e-6)                                          # hindsight key of s (feature space)
                if A.linrevctl: kh = fw   # control: same projections, s addressed by its own write features (no hindsight)
                R = (fq @ kh.transpose(-1, -2)) * tri
                y = y + ((R @ u) / R.sum(-1, keepdim=True).clamp(min=1e-6)).to(vt.dtype)
        elif bias is not None:
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias)
        elif A.nullslot:
            _T = kt.size(2); kc = torch.cat([s.kn0.to(kt.dtype).view(1, s.h, 1, s.hd).expand(B, -1, 1, -1), kt], 2); vc = torch.cat([torch.zeros_like(vt[:, :, :1]), vt], 2)
            y = F.scaled_dot_product_attention(qt, kc, vc, attn_mask=torch.ones(_T, _T + 1, device=x.device, dtype=torch.bool).tril(1))
        else:
            if ATT_MASK[0] is not None and ATT_MASK[0].shape[0] == qt.size(0) and ATT_MASK[0].shape[-1] == kt.size(2): y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=ATT_MASK[0])   # packm: the second fact document cannot see the first
            elif WIN[0]: _T = kt.size(2); _i = torch.arange(_T, device=x.device); y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=(_i[None, :] <= _i[:, None]) & (_i[:, None] - _i[None, :] < WIN[0]))   # --win: local window only, so in-context copying is off for this batch
            else: y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=not A.mdlm)   # --mdlm: bidirectional (masked diffusion LM)
        if ngp is not None:
            if A.ngrep:   # uniform over matched positions, values of the same heads
                yr = ngp.to(vt.dtype) @ vt[:, :2]
            else:         # content-selected among matched positions; no match -> zero
                sc = (qt[:, :2] @ kt[:, :2].transpose(-1, -2)) / s.hd ** 0.5
                sc = sc.float().masked_fill(~ngp, float("-inf")); has = ngp.any(-1, keepdim=True)
                pr = torch.where(has, sc.softmax(-1), torch.zeros_like(sc)).nan_to_num(0.0)
                yr = pr.to(vt.dtype) @ vt[:, :2]
            y = torch.cat([yr, y[:, 2:]], 1)
        if A.succ:   # keys [null, k_0..k_{T-2}] paired with values [0, v_1..v_{T-1}]: causal position t reads up to k_{t-1} -> v_t (no future)
            kn = rms(s.knull).to(kt.dtype).view(1, s.h, 1, s.hd).expand(B, -1, 1, -1)
            ks = torch.cat([kn, kt[:, :, :-1]], 2); vs = torch.cat([torch.zeros_like(vt[:, :, :1]), vt[:, :, 1:]], 2)
            ys = F.scaled_dot_product_attention(qt, ks, vs, is_causal=True)
            ms = torch.sigmoid(s.smix).view(1, -1, 1, 1).to(y.dtype); y = (1 - ms) * y + ms * ys
        if A.revise2:   # token s writes u_s addressed by its hindsight key K'_s = sum_{j<s} b_sj k_j (keys of what s attends to); later queries read it
            qw = s.rot(rms(s.qw(x).view(B, T, s.h, s.hd))).transpose(1, 2); u = s.uw(x).view(B, T, s.h, s.hd).transpose(1, 2)
            kz = torch.cat([torch.zeros_like(kt[:, :, :1]), kt[:, :, :-1]], 2)                  # strictly earlier keys (slot 0 = null)
            kh = F.scaled_dot_product_attention(qw, kz, kz, is_causal=True)                    # hindsight key of each position
            y = y + F.scaled_dot_product_attention(qt, rms(kh), u, is_causal=True)
        if A.revise or A.revisectl:
            qw = s.rot(rms(s.qw(x).view(B, T, s.h, s.hd))).transpose(1, 2); u = s.uw(x).view(B, T, s.h, s.hd).transpose(1, 2)
            if A.revisectl:   # control: the same projections as one more plain causal attention path
                y = y + F.scaled_dot_product_attention(qw, kt, u, is_causal=True)
            else:
                sc = 1.0 / s.hd ** 0.5; ar = torch.arange(T, device=x.device)
                Am = ((qt @ kt.transpose(-1, -2)).float() * sc).masked_fill(ar.view(1, -1) > ar.view(-1, 1), float("-inf")).softmax(-1)      # a_tj, j <= t
                Bm = ((qw @ kt.transpose(-1, -2)).float() * sc).masked_fill(ar.view(1, -1) >= ar.view(-1, 1), float("-inf")).softmax(-1).nan_to_num(0.0)   # b_sj, j < s
                Cm = (Am @ Bm.transpose(-1, -2)).masked_fill(ar.view(1, -1) > ar.view(-1, 1), 0.0)   # c_ts = sum_j a_tj b_sj, s <= t
                y = y + (Cm.to(u.dtype) @ u)
        if A.attnfloor > 0:   # gradient-only responsibility floor: forward unchanged, every earlier value receives eps/n of gradient
            cnt = torch.arange(1, T + 1, device=x.device, dtype=vt.dtype).view(1, 1, T, 1)
            um = vt.cumsum(2) / cnt
            y = y + A.attnfloor * (um - um.detach())
        if A.twotemp:
            m = torch.sigmoid(s.mixlog).view(1, -1, 1, 1).to(y.dtype)
            y = (1 - m) * y + m * F.scaled_dot_product_attention(qt * 4.0, kt, vt, is_causal=True)
        if A.ablate and ABL_ON[0] and s.li == int(A.ablate.split(",")[0]):
            _keep = torch.ones(s.h, device=y.device, dtype=y.dtype); _keep[int(A.ablate.split(",")[1])] = 0; y = y * _keep.view(1, -1, 1, 1)
        if A.soft_layers and str(s.li) in A.soft_layers.split(',') and (A.soft_off or SOFT_OFF[0]): y = y * 0   # hybrid: the softmax layers' attention removed (--soft_off at evaluation, --soft_drop during training)
        if A.oread > 0 and s.training:
            for _L, _H in OREAD_ACT[0]:
                if _L == s.li: OREAD_O[0].append(y[:, _H] @ s.o.weight[:, _H * s.hd:(_H + 1) * s.hd].t())   # this head's contribution to the residual stream
        _solo = [_H for _L, _H in SOLO_ACT[0] if _L == s.li] if s.training else []
        if _solo:   # --solo_spec: these heads' contributions reach the residual stream without gradient, so the LM loss does not train them (only the shared readout does)
            _m = torch.zeros(s.h, device=y.device, dtype=y.dtype); _m[_solo] = 1; _m = _m.view(1, -1, 1, 1)
            _ys = (y * _m).transpose(1, 2).reshape(B, T, d); y = (y * (1 - _m)).transpose(1, 2).reshape(B, T, d)
            if A.gate: _g = torch.sigmoid(s.g(x)); y = y * _g; _ys = _ys * _g
            _os = s.o(_ys)
            if A.solo_copy and SOLO_MASK[0] is not None: return s.o(y) + torch.where(SOLO_MASK[0], _os, _os.detach()), v1
            return s.o(y) + _os.detach(), v1
        y = y.transpose(1, 2).reshape(B, T, d)
        if A.gate: y = y * torch.sigmoid(s.g(x))
        return s.o(y), v1


ROUTE = [None]   # (B,T) bool: second FFN expert for these positions (set per forward when --novroute / --hashroute)
class MLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.fc = nn.Linear(d, 4 * d, bias=False); s.pr = nn.Linear(4 * d, d, bias=False); nn.init.zeros_(s.pr.weight)
        if A.novroute or A.hashroute: s.fc2 = nn.Linear(d, 4 * d, bias=False); s.pr2 = nn.Linear(4 * d, d, bias=False); nn.init.zeros_(s.pr2.weight)
    def forward(s, x):
        y = s.pr(F.relu(s.fc(x)).square())
        if ROUTE[0] is None: return y
        return torch.where(ROUTE[0].unsqueeze(-1), s.pr2(F.relu(s.fc2(x)).square()), y)


class Block(nn.Module):
    def __init__(s, d, h, T, li):
        super().__init__(); s.attn = Attn(d, h, T, li); s.mlp = MLP(d)
        if A.ngramhead:
            s.ngn = 3 if A.ng3 else 2; s.ngv = nn.Linear(d, d, bias=False); s.ngo = nn.Linear(s.ngn * d, d, bias=False); nn.init.zeros_(s.ngo.weight)
            if A.ngrec: s.ngdec = nn.Parameter(torch.full((s.ngn,), -4.0))   # log decay rate per n: weight exp(-softplus(r) * distance)
        if A.surprise: s.sp = nn.Parameter(torch.tensor([0.0, 3.0, 0.0, 3.0]))   # (slope, bias) for attn and mlp updates
        if A.momentum: s.beta = nn.Parameter(torch.zeros(2))
        if A.canon: s.ca, s.cc = Canon(d), Canon(d)
    def attn_in(s, x): x = rms(x); return s.ca(x) if A.canon else x
    def mlp_in(s, x): x = rms(x); return s.cc(x) if A.canon else x


class GPT(nn.Module):
    def __init__(s):
        super().__init__(); d, L = A.d, A.layers; s._wanth = False
        s.emb = nn.Embedding(V_SIZE, d); s.blocks = nn.ModuleList(Block(d, A.heads, A.seq, i) for i in range(L))
        s.head = nn.Linear(d, V_SIZE, bias=False); nn.init.normal_(s.head.weight, std=0.02); nn.init.normal_(s.emb.weight, std=1.0)
        if A.attnres: s.rq = nn.Parameter(torch.zeros(2 * L + 1, d))   # pseudo-queries: 2 per block + final
        if A.surpfeat: s.semb = nn.Linear(1, d, bias=False); nn.init.zeros_(s.semb.weight)   # surprisal (nats) -> stream
        if A.cachesoft: s.cth = nn.Parameter(torch.tensor(1.0))
        if A.cgfeat: s.cgf = nn.Parameter(torch.zeros(2))
        if A.attnptr: s.apg = nn.Linear(d, 1)
        if A.cachepoe: s.pbeta = nn.Parameter(torch.tensor(0.0))   # logits += beta * log(eps + p_cache)
        if A.readmix: s.rmx = nn.Parameter(torch.zeros(A.layers))
        if A.recur: s.rg = nn.Parameter(torch.tensor([0.0, 0.0]))   # gate on mid-layer predictive entropy: sigmoid(a * (H - mean H) + b)
        if A.psguide:   # prediction-state guidance for the last block's attention: low-rank similarity of penultimate states
            s.pgq = nn.Linear(d, 64, bias=False); s.pgk = nn.Linear(d, 64, bias=False); s.pga = nn.Parameter(torch.zeros(A.heads))
        if A.copyrun: s.crun = nn.Embedding(17, d); nn.init.zeros_(s.crun.weight)        # run length of correct 1-gram hints (0..15, 16 = no hint)
        if A.hinterr: s.herr = nn.Linear(d, d, bias=False); nn.init.zeros_(s.herr.weight)   # vector error between the token and the previous hint
        if A.hintdist: s.hd1 = nn.Linear(d, d, bias=False); s.hd2 = nn.Linear(d, d, bias=False); nn.init.zeros_(s.hd1.weight); nn.init.zeros_(s.hd2.weight)
        if A.hint34: s.h3 = nn.Embedding(V_SIZE + 1, d); s.h4 = nn.Embedding(V_SIZE + 1, d); nn.init.zeros_(s.h3.weight); nn.init.zeros_(s.h4.weight)
        if A.hints: s.h1 = nn.Embedding(V_SIZE + 1, d); s.h2 = nn.Embedding(V_SIZE + 1, d); nn.init.zeros_(s.h1.weight); nn.init.zeros_(s.h2.weight)   # gate weights on cache confidence: [max weight, -entropy]
        if A.cacheproj:
            CH, CR = A.cheads, A.crank
            s.cq = nn.Linear(d, CH * CR, bias=False); s.ck = nn.Linear(d, CH * CR, bias=False); s.cmix = nn.Parameter(torch.zeros(CH))
            with torch.no_grad():   # start each head at a raw-cosine cache on its own slice of the hidden state (identity-like projections)
                s.cq.weight.zero_(); s.ck.weight.zero_()
                for hh in range(CH):
                    for r in range(CR):
                        c = (hh * CR + r) % d
                        s.cq.weight[hh * CR + r, c] = 1.0; s.ck.weight[hh * CR + r, c] = 1.0
        if A.creadmid >= 0:   # mid-network read-back: prediction-state retrieval of the tokens that followed similar states, read by the upper blocks
            s.mq = nn.Linear(d, 256, bias=False); s.mk = nn.Linear(d, 256, bias=False); s.mo = nn.Linear(d, d, bias=False); s.mth = nn.Parameter(torch.tensor(1.0)); nn.init.zeros_(s.mo.weight)
            with torch.no_grad():
                s.mq.weight.zero_(); s.mk.weight.zero_()
                for r in range(256): s.mq.weight[r, r % d] = 1.0; s.mk.weight[r, r % d] = 1.0
        if A.hind:   # hindsight attention: second pass, the input layer reads the first pass's top states of strictly earlier positions
            s.hq = nn.Linear(d, d, bias=False); s.hk = nn.Linear(d, d, bias=False); s.hv = nn.Linear(d, d, bias=False); s.ho = nn.Linear(d, d, bias=False); nn.init.zeros_(s.ho.weight)
        if A.vthink >= 0:   # vocabulary-thought layer after block vthink: state -> softmax over the vocabulary (output head) -> expected word embedding -> stream
            s.vo = nn.Linear(d, d, bias=False); nn.init.zeros_(s.vo.weight); s.vt = nn.Parameter(torch.tensor(0.0))
            if A.vuntied: s.vmem = nn.Parameter(torch.randn(V_SIZE, d) * 0.02)   # control: same-size softmax memory not tied to the vocabulary
        if A.cread: s.crp = nn.Linear(d, d, bias=False); nn.init.zeros_(s.crp.weight)   # cache read-back: retrieved output embeddings -> final state
        if A.semhead or A.mos2:   # third output expert: semantic copy (neighbours of the copy head's tokens) or, as control, a second LM softmax
            s.cg2 = nn.Linear(d, 1); nn.init.zeros_(s.cg2.weight); nn.init.constant_(s.cg2.bias, -3.0); s.stau = nn.Parameter(torch.tensor(0.0))
            if A.mos2: s.mosp = nn.Linear(d, d, bias=False)
        if A.cpstyle:   # Copy-Paste-LM-style head: gate = probability of an extra <copy> slot competing with the vocabulary; learnable sink key that rolls mass back to the LM
            s.cpl = nn.Linear(d, 1); nn.init.zeros_(s.cpl.weight); nn.init.constant_(s.cpl.bias, -3.0); s.csink = nn.Parameter(torch.zeros(A.cheads))
        if A.mosk: s.mosproj = nn.Linear(d, A.mosk * d, bias=False); s.mosgate = nn.Linear(d, A.mosk)   # mixture of softmaxes (Yang et al. 2018)
        if A.expert: s.eg = nn.Linear(d, 1); nn.init.zeros_(s.eg.weight); nn.init.constant_(s.eg.bias, -3.0)   # gate of a frozen external expert LM
        if A.cachelm: s.cg = nn.Linear(d, 2 if A.cache2 else 1); nn.init.zeros_(s.cg.weight); nn.init.constant_(s.cg.bias, -3.0)
        if A.depthemb:
            s.register_buffer("ddelta", torch.from_numpy(__import__("numpy").load("./data/depthdelta.npy")).long(), persistent=False)
            s.demb = nn.Embedding(16, d); nn.init.zeros_(s.demb.weight)
        if A.segbias: s.register_buffer("bnd", torch.from_numpy(__import__("numpy").load("./data/boundary.npy")), persistent=False)
        if A.pairaff: s.pu = nn.Embedding(V_SIZE, 16); s.pv = nn.Embedding(V_SIZE, 16); nn.init.normal_(s.pu.weight, std=0.25); nn.init.normal_(s.pv.weight, std=0.25)
    def mid_read(s, x, idx):
        B, T, _ = x.shape; hn = rms(x); hk = F.pad(hn, (0, 0, 1, 0))[:, :-1]
        q = (s.mq(hn) * (s.mth.exp() / 8.0)).view(B, T, 4, 64).transpose(1, 2); k = s.mk(hk).view(B, T, 4, 64).transpose(1, 2)
        v = (hn if A.midctl else s.head.weight[idx]).to(q.dtype).unsqueeze(1).expand(-1, 4, -1, -1)   # control: plain hidden-state values
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0).mean(1)
        return s.mo(o)
    def mix(s, hist, i):   # softmax over depth history (Attention Residuals)
        Hs = torch.stack(hist, 0)                                   # n,B,T,d
        w = torch.einsum("nbtd,d->nbt", rms(Hs), s.rq[i].to(Hs.dtype)).softmax(0)
        return torch.einsum("nbt,nbtd->btd", w, Hs)
    def forward(s, idx, tgt=None):
        x = s.emb(idx); v1 = None
        if A.solo_copy and tgt is not None and s.training:   # induction-copyable positions (bigram seen before): the solo heads keep LM-loss gradient only there
            _T = idx.size(1); SOLO_MASK[0] = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(_T, _T, device=idx.device, dtype=torch.bool), -1)).any(-1).unsqueeze(-1)
        if A.novroute:   # repeat regime: token x_t already occurred earlier in the context (exact, causal)
            ROUTE[0] = ((idx.unsqueeze(2) == idx.unsqueeze(1)) & torch.tril(torch.ones(idx.size(1), idx.size(1), device=idx.device, dtype=torch.bool), -1)).any(-1)
        elif A.hashroute: ROUTE[0] = (idx % 2 == 1)   # control: same experts, routing by token-id parity
        if A.hints:   # most recent earlier occurrence of token t (1-gram) / of tokens (t-1, t) (2-gram): embed the token that followed it
            Bh, Th = idx.shape; ar = torch.arange(Th, device=idx.device)
            prevtok = F.pad(idx, (1, 0), value=-1)[:, :-1]
            m1 = (idx.unsqueeze(2) == prevtok.unsqueeze(1)) & (ar.view(1, 1, -1) <= ar.view(1, -1, 1))          # j<=t with x_{j-1} == x_t
            prev2 = F.pad(idx, (2, 0), value=-2)[:, :-2]; cur1 = F.pad(idx, (1, 0), value=-3)[:, :-1]
            m2 = m1 & (cur1.unsqueeze(2) == prev2.unsqueeze(1))
            def last_next(mm):   # token x_j at the largest matching j, or V_SIZE (= none)
                pos = torch.where(mm, ar.view(1, 1, -1), torch.full_like(ar.view(1, 1, -1), -1)).amax(-1)
                return torch.where(pos >= 0, idx.gather(1, pos.clamp(min=0)), torch.full_like(pos, V_SIZE))
            hint1 = last_next(m1); x = x + s.h1(hint1) + s.h2(last_next(m2))
            if A.hintdist:   # count-weighted mean input embedding of every token that followed a 1-/2-gram match
                E0 = s.emb(idx)
                for mm, lin in ((m1, s.hd1), (m2, s.hd2)):
                    wq = mm.float(); wq = wq / wq.sum(-1, keepdim=True).clamp(min=1.0)
                    x = x + lin(wq.to(E0.dtype) @ E0)
            if A.hint34:
                prev3 = F.pad(idx, (3, 0), value=-4)[:, :-3]; cur2 = F.pad(idx, (2, 0), value=-5)[:, :-2]
                m3 = m2 & (cur2.unsqueeze(2) == prev3.unsqueeze(1))
                prev4 = F.pad(idx, (4, 0), value=-6)[:, :-4]; cur3 = F.pad(idx, (3, 0), value=-7)[:, :-3]
                m4 = m3 & (cur3.unsqueeze(2) == prev4.unsqueeze(1))
                x = x + s.h3(last_next(m3)) + s.h4(last_next(m4))
            if A.copyrun or A.hinterr:
                prevhint = F.pad(hint1, (1, 0), value=V_SIZE)[:, :-1]                       # hint made at t-1 for token t
                ok = (prevhint == idx)
                if A.copyrun:   # consecutive correct hints ending at t (causal: uses tokens <= t)
                    okf = ok.long(); cs = okf.cumsum(1)
                    last_bad = torch.where(~ok, torch.arange(Th, device=idx.device).view(1, -1), torch.full_like(okf, -1)).cummax(1).values
                    run = (torch.arange(Th, device=idx.device).view(1, -1) - last_bad).clamp(max=15)
                    run = torch.where(prevhint == V_SIZE, torch.full_like(run, 16), run)
                    x = x + s.crun(run)
                if A.hinterr:
                    has_h = (prevhint != V_SIZE).unsqueeze(-1).to(x.dtype)
                    x = x + s.herr(rms(s.emb(idx) - s.emb(prevhint.clamp(max=V_SIZE - 1)))) * has_h
        if A.depthemb:
            dep = F.pad(s.ddelta[idx].cumsum(1), (1, 0))[:, :-1].clamp(0, 15)   # depth before token t (causal)
            x = x + s.demb(dep)
        if A.hind and not s._wanth:
            s._wanth = True; Hp = s.forward(idx); s._wanth = False
            B_, T_, d_ = x.shape; nh = A.heads; Hs = F.pad(rms(Hp), (0, 0, 1, 0))[:, :-1]   # position j holds top state of j-1 (row 0 = null)
            q = s.hq(rms(x)).view(B_, T_, nh, -1).transpose(1, 2); k = s.hk(Hs).view(B_, T_, nh, -1).transpose(1, 2); v = s.hv(Hs).view(B_, T_, nh, -1).transpose(1, 2)
            x = x + s.ho(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T_, d_))
        if A.attnres:
            hist = [x]
            for j, b in enumerate(s.blocks):
                h = s.mix(hist, 2 * j); a, v1 = b.attn(b.attn_in(h), v1); hist.append(a)
                h = s.mix(hist, 2 * j + 1); hist.append(b.mlp(b.mlp_in(h)))
            x = s.mix(hist, 2 * len(s.blocks))
        else:
            up = None; T = idx.size(1); L = len(s.blocks); hcache = None
            causal = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
            match = None; surp = None
            if A.induct or A.lexmatch or A.ngramhead or A.ngrep or A.ngsel:   # match[t, j] = 1 if token[j-1] == token[t] (j >= 1, j <= t)
                prev = F.pad(idx, (1, 0), value=-1)[:, :-1]
                match = (idx.unsqueeze(2) == prev.unsqueeze(1)).float().unsqueeze(1)          # B,1,T,T
            if A.lexmatch or A.ngramhead or A.lex5 or A.ngrep or A.ngsel or A.ngback:
                same = (idx.unsqueeze(2) == idx.unsqueeze(1)).float().unsqueeze(1)
                prev2 = F.pad(idx, (2, 0), value=-2)[:, :-2]; cur1 = F.pad(idx, (1, 0), value=-3)[:, :-1]
                match2 = match * (cur1.unsqueeze(2) == prev2.unsqueeze(1)).float().unsqueeze(1)   # also token[j-2] == token[t-1]
                if A.lex5 or A.ng3 or A.ngback:
                    prev3 = F.pad(idx, (3, 0), value=-4)[:, :-3]; cur2 = F.pad(idx, (2, 0), value=-5)[:, :-2]
                    match3 = match2 * (cur2.unsqueeze(2) == prev3.unsqueeze(1)).float().unsqueeze(1)
                    sameprev = (cur1.unsqueeze(2) == prev.unsqueeze(1)).float().unsqueeze(1)      # token[j-1] == token[t-1]
            if A.segbias:   # sentence id = number of boundary tokens strictly before the position
                sid = F.pad(s.bnd[idx].long().cumsum(1), (1, 0))[:, :-1]
                sameseg = (sid.unsqueeze(2) == sid.unsqueeze(1)).float().unsqueeze(1)
            if A.countsal:  # occurrences of token j at positions <= j
                eq = (idx.unsqueeze(2) == idx.unsqueeze(1)).float()
                cnt = torch.log1p((eq * torch.tril(torch.ones(T, T, device=idx.device))).sum(-1) - 1).unsqueeze(1).unsqueeze(2)   # B,1,1,T
            if A.pairaff:   # learned low-rank token-pair affinity U[x_t] . V[x_j]
                paff = (s.pu(idx).float() @ s.pv(idx).float().transpose(1, 2)).unsqueeze(1) / 4.0
            if A.softmatch:   # cosine of learned input embeddings: soft (semantic) versions of same-token and induction matches
                en = F.normalize(s.emb(idx).float(), dim=-1); enp = F.pad(en, (0, 0, 1, 0))[:, :-1]
                ssame = (en @ en.transpose(1, 2)).unsqueeze(1); sind = (en @ enp.transpose(1, 2)).unsqueeze(1)
            ngp = None
            if A.ngrep or A.ngsel:
                tri = torch.tril(torch.ones(T, T, device=idx.device))
                if A.ngback:   # longest available order wins (Katz-style back-off), per query row
                    m3, m2, m1 = match3 * tri, match2 * tri, match * tri
                    h3 = m3.sum(-1, keepdim=True) > 0; h2 = m2.sum(-1, keepdim=True) > 0
                    mb = torch.where(h3, m3, torch.where(h2, m2, m1))
                    mm = torch.cat([mb, m1], 1)
                else:
                    mm = torch.cat([match, match2], 1) * tri                     # B,2,T,T
                ngp = mm / mm.sum(-1, keepdim=True).clamp(min=1.0) if A.ngrep else mm.bool()
            for li, b in enumerate(s.blocks):
                bias = None
                if A.psguide and li == L - 1:   # bias_tj = alpha_h * log softmax_j(q_t . k_{j}) over j <= t, from the penultimate residual stream
                    hq = rms(x); qg = s.pgq(hq).float(); kg = s.pgk(hq).float()
                    sg = (qg @ kg.transpose(1, 2)) / 8.0
                    sg = sg.masked_fill(torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1), float("-inf"))
                    bias = (s.pga.view(1, -1, 1, 1) * sg.log_softmax(-1).unsqueeze(1).clamp(min=-30)).to(x.dtype)
                    bias = bias + torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1).view(1, 1, T, T).to(x.dtype)
                if A.induct or A.lexmatch or A.softmatch or A.segbias or A.countsal or A.pairaff or (A.surpfeat and surp is not None):
                    bias = causal.view(1, 1, T, T).to(x.dtype)
                    if A.lexmatch:
                        lw = b.attn.lex.to(x.dtype)
                        bias = bias + lw[0].view(1, -1, 1, 1) * same.to(x.dtype) + lw[1].view(1, -1, 1, 1) * match.to(x.dtype) + lw[2].view(1, -1, 1, 1) * match2.to(x.dtype)
                        if A.lex5: bias = bias + lw[3].view(1, -1, 1, 1) * match3.to(x.dtype) + lw[4].view(1, -1, 1, 1) * sameprev.to(x.dtype)
                    if A.segbias: bias = bias + b.attn.segw.view(1, -1, 1, 1).to(x.dtype) * sameseg.to(x.dtype)
                    if A.countsal: bias = bias + b.attn.cntw.view(1, -1, 1, 1).to(x.dtype) * cnt.to(x.dtype)
                    if A.pairaff: bias = bias + b.attn.pairw.view(1, -1, 1, 1).to(x.dtype) * paff.to(x.dtype)
                    if A.softmatch:
                        sw = b.attn.soft.to(x.dtype)
                        bias = bias + sw[0].view(1, -1, 1, 1) * ssame.to(x.dtype) + sw[1].view(1, -1, 1, 1) * sind.to(x.dtype)
                    if A.induct: bias = bias + b.attn.ind.view(1, -1, 1, 1).to(x.dtype) * match.to(x.dtype)
                    if A.surpfeat and surp is not None: bias = bias + b.attn.sw.view(1, -1, 1, 1).to(x.dtype) * surp.view(surp.size(0), 1, 1, T).to(x.dtype)
                for kind in (0, 1):
                    if kind == 0:
                        u, v1 = b.attn(b.attn_in(x), v1, bias, ngp)
                        if A.ngramhead:   # hard n-gram heads: uniform average of value(j) over causal positions whose previous n tokens match
                            tri = torch.tril(torch.ones(T, T, device=x.device))
                            vv = b.ngv(rms(x)); outs = []
                            if A.ngrec: dist = (torch.arange(T, device=x.device).view(T, 1) - torch.arange(T, device=x.device).view(1, T)).clamp(min=0).float()
                            for ni, mm in enumerate((match, match2, match3) if A.ng3 else (match, match2)):
                                w = mm.squeeze(1) * tri
                                if A.ngrec: w = w * torch.exp(-F.softplus(b.ngdec[ni]) * dist)
                                w = w / w.sum(-1, keepdim=True).clamp(min=1e-6 if A.ngrec else 1.0)
                                outs.append(w.to(vv.dtype) @ vv)
                            u = u + b.ngo(torch.cat(outs, -1))
                    else: u = b.mlp(b.mlp_in(x))
                    if A.surprise:   # update strength set by its agreement with the stream (cosine), per token
                        c = F.cosine_similarity(u.float(), x.float(), dim=-1, eps=1e-6).unsqueeze(-1).to(u.dtype)
                        u = u * torch.sigmoid(b.sp[2 * kind] * c + b.sp[2 * kind + 1]) / torch.sigmoid(b.sp[2 * kind + 1].detach())
                    x = x + u + (b.beta[kind] * up if (A.momentum and up is not None) else 0)
                    up = u
                if A.readmix: rmacc = (rmacc if li > 0 else 0) + s.rmx[li] * rms(x)
                if A.recurgate and li == L // 2 - 1: hmid = x
                if A.ckeylayer == li: hcache = x
                if A.creadmid == li: x = x + s.mid_read(x, idx)
                if A.vthink == li:
                    Wv = s.vmem if A.vuntied else s.head.weight
                    pv = (F.linear(rms(x), Wv.to(x.dtype)) * s.vt.exp()).softmax(-1)
                    x = x + s.vo(pv @ Wv.to(x.dtype))
                if A.surpfeat and li == L // 2 - 1:   # early-exit surprisal of each token given its past (causal), fed to upper layers
                    with torch.no_grad():
                        lg = s.head(rms(x)).float(); lp = lg.log_softmax(-1)
                        sv = -lp[:, :-1].gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)          # surprisal of token t+1 given <= t
                        surp = F.pad(sv, (1, 0), value=float(sv.mean()))                         # token t's surprisal (first token: mean)
                        surp = (surp - surp.mean()) / (surp.std() + 1e-6)
                    x = x + s.semb(surp.unsqueeze(-1).to(x.dtype))
        if A.recur:   # one extra, weight-tied pass of the last block; with --recurgate its update is scaled per token by uncertainty
            bL = s.blocks[-1]
            ua, _ = bL.attn(bL.attn_in(x), None); x2 = x + ua; x2 = x2 + bL.mlp(bL.mlp_in(x2)); upd = x2 - x
            if A.recurgate:
                with torch.no_grad():
                    lpm = s.head(rms(hmid)).float().log_softmax(-1); Hm = -(lpm.exp() * lpm).sum(-1)
                    Hm = Hm / 5.0 - 1.0   # fixed affine scaling (no batch statistics: they would leak future positions)
                gate = torch.sigmoid(s.rg[0] * Hm + s.rg[1]).unsqueeze(-1).to(x.dtype) * 2.0
                upd = upd * gate
            x = x + upd
        if A.hind and s._wanth: return x
        hout = rms(x) + rmacc if A.readmix else rms(x)
        logits = s.head(hout).float(); logits = 15 * torch.tanh(logits / 15)
        xc = x.detach() if A.cstopgrad else x   # stop-gradient ablation: the cache path cannot shape the backbone
        if A.cachelm:   # p_cache(t) = distribution of tokens that followed earlier occurrences of token t (positions j <= t)
            Bc, Tc = idx.shape; prev = F.pad(idx, (1, 0), value=-1)[:, :-1]
            if A.cachesoft:   # position j (>=1) holds token x_j; its key is the hidden state at j-1 (which predicted x_j)
                hsrc = hcache if (A.ckeylayer >= 0 and hcache is not None) else xc
                hn = rms(hsrc) if A.cachebf else rms(hsrc).float(); hk = F.pad(hn, (0, 0, 1, 0))[:, :-1]
                allow = torch.tril(torch.ones(Tc, Tc, device=idx.device, dtype=torch.bool)); allow[:, 0] = False
                if A.cacheproj:
                    CH, CR = A.cheads, A.crank
                    qh = s.cq(hn).view(Bc, Tc, CH, CR).transpose(1, 2); kh = s.ck(hk).view(Bc, Tc, CH, CR).transpose(1, 2)
                    sc = (s.cth.exp() * (qh @ kh.transpose(-1, -2)) / CR ** 0.5).float()
                    if A.cpstyle:   # sink column first: its mass a is handed back to the LM
                        full = torch.cat([s.csink.float().view(1, CH, 1, 1).expand(Bc, CH, Tc, 1), sc.masked_fill(~allow, float("-inf"))], -1).softmax(-1)
                        wh = full[..., 1:]; cpa = (s.cmix.softmax(0).view(1, CH, 1) * full[..., 0]).sum(1)          # B,T
                    else:
                        wh = sc.masked_fill(~allow, float("-inf")).softmax(-1).nan_to_num(0.0)        # B,CH,T,T
                    w = (s.cmix.softmax(0).view(1, CH, 1, 1) * wh).sum(1)
                else:
                    sc = s.cth.exp() * (hn @ hk.transpose(1, 2)) / hn.size(-1) ** 0.5
                    w = sc.masked_fill(~allow, float("-inf")).softmax(-1).nan_to_num(0.0)
                has = allow.any(-1, keepdim=True).unsqueeze(0).expand(Bc, -1, -1)
            else:
                w = (idx.unsqueeze(2) == prev.unsqueeze(1)).float() * torch.tril(torch.ones(Tc, Tc, device=idx.device))
                has = w.sum(-1, keepdim=True) > 0; w = w / w.sum(-1, keepdim=True).clamp(min=1.0)
            if A.cread:   # feed the cache's proposal (weighted output embeddings of the retrieved tokens) back into the final state
                rv = torch.bmm(w.to(s.head.weight.dtype), rms(x).to(s.head.weight.dtype) if A.creadctl else s.head.weight[idx]); x = x + s.crp(rv.to(x.dtype))   # control: retrieved hidden states
                logits = s.head(rms(x)).float(); logits = 15 * torch.tanh(logits / 15)
            gl = s.cg(rms(xc)).float()
            if A.cgfeat and A.cachesoft:
                wmax = w.max(-1).values; went = -(w * (w + 1e-9).log()).sum(-1)
                gl = gl + (s.cgf[0] * wmax - s.cgf[1] * went / math.log(Tc)).unsqueeze(-1)
            lam = torch.sigmoid(gl[..., :1]) * has.float()
            if A.cpstyle:   # lambda = softmax mass of <copy> against the vocabulary logits; effective copy weight lambda * (1 - sink)
                lcp = s.cpl(rms(x)).float().squeeze(-1); lam_cp = torch.sigmoid(lcp - torch.logsumexp(logits, -1))
                lam = (lam_cp * (1 - cpa)).unsqueeze(-1)
            if A.cacheflash and tgt is not None:   # exact, no dense T x T: -log p = LSE_all - LSE_match per head, mixed over heads
                hn2 = rms(x); B2, T2, _ = hn2.shape; scl = float(s.cth.exp().item()) if False else None
                qh = s.cq(hn2).view(B2, T2, 4, 64).transpose(1, 2)
                kh = s.ck(F.pad(hn2, (0, 0, 1, 0))[:, :-1]).view(B2, T2, 4, 64).transpose(1, 2)      # key at j = ck(h_{j-1}); j=0 -> ck(0) = 0 (sentinel)
                qs = qh * (s.cth.exp() / 8.0).to(qh.dtype)                                          # fold the learned temperature into q
                lse = FlashLSE.apply(qs.to(torch.bfloat16).contiguous(), kh.to(torch.bfloat16).contiguous(), 1.0)   # B,4,T incl. sentinel exp(0)=1
                lse_all = torch.log(torch.expm1(lse.clamp(min=1e-6)).clamp(min=1e-30))              # remove the sentinel term exactly
                mt = (idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(T2, T2, device=idx.device, dtype=torch.bool))
                mt[:, :, 0] = False; bi, ti, ji = mt.nonzero(as_tuple=True)                          # pairs: token x_j == y_t, 1 <= j <= t
                sp = (qs[bi, :, ti].float() * kh[bi, :, ji].float()).sum(-1)                         # P,4
                flat = bi * T2 + ti; n = B2 * T2
                mx = torch.full((n, 4), float("-inf"), device=idx.device).scatter_reduce(0, flat.unsqueeze(1).expand(-1, 4), sp, "amax")
                se = torch.zeros((n, 4), device=idx.device).index_add_(0, flat, torch.exp(sp - mx[flat]))
                hasm = torch.isfinite(mx)
                lse_m = torch.where(hasm, mx, torch.zeros_like(mx)) + torch.log(se.clamp(min=1e-30))
                lse_m = torch.where(hasm, lse_m, torch.full_like(lse_m, -1e4)).view(B2, T2, 4).permute(0, 2, 1)   # finite where no match
                logp_h = lse_m - lse_all                                                             # B,4,T
                logp = torch.logsumexp(s.cmix.log_softmax(0).view(1, 4, 1) + logp_h, 1)            # B,T
                lpy = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)   # fused, no BxTxV log_softmax
                lm = lam.squeeze(-1)
                mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lm + 1e-9) + logp)
                return -mix.mean()
            if A.maxmix and tgt is not None:   # p(y) = max(p_LM(y), lam p_copy(y)) / Z, Z = 1 + sum over distinct context tokens of relu(lam p_copy - p_LM): no tax where the experts agree
                eqm = (idx.unsqueeze(2) == idx.unsqueeze(1)).float()
                Pj = torch.bmm(w, eqm)                                                   # p_copy(x_j) seen from row t
                tri = torch.tril(torch.ones(Tc, Tc, device=idx.device)).expand(Bc, -1, -1)
                cnt = torch.bmm(tri, eqm)                                                # occurrences of x_j among j' <= t
                lse = torch.logsumexp(logits, -1)
                pl = torch.exp(logits.gather(-1, idx.unsqueeze(1).expand(Bc, Tc, Tc)) - lse.unsqueeze(-1))   # p_LM(x_j)
                lmb = lam.squeeze(-1).unsqueeze(-1)
                corr = (F.relu(lmb * Pj - pl) * tri / cnt.clamp(min=1.0)).sum(-1)
                lpy = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lse
                py = (w * (idx.unsqueeze(1) == tgt.unsqueeze(2)).float()).sum(-1)
                lmax = torch.maximum(lpy, torch.log(lam.squeeze(-1) * py + 1e-9))
                out = -(lmax - torch.log1p(corr)).mean()
                if s.training and A.lmaux > 0: out = out - A.lmaux * lpy.mean()
                return out
            if A.cachepoe and A.poefast and tgt is not None:   # exact PoE + mixture loss without any B x T x V cache tensor
                eps = 1e-3; beta = F.softplus(s.pbeta)
                eqm = (idx.unsqueeze(2) == idx.unsqueeze(1)).float()                       # [x_i == x_j]
                Pj = torch.bmm(w, eqm)                                                     # p_cache(x_j) seen from row t
                cnt = torch.bmm((w > 0).float() * 0 + torch.tril(torch.ones(Tc, Tc, device=idx.device)).expand(Bc, -1, -1), eqm)   # occurrences of x_j among keys j' <= t
                valid = (torch.tril(torch.ones(Tc, Tc, device=idx.device, dtype=torch.bool)).unsqueeze(0) & (Pj > 0))
                lse = torch.logsumexp(logits, -1)                                          # LSE of the LM logits (B,T)
                lj = logits.gather(-1, idx.unsqueeze(1).expand(Bc, Tc, Tc))                # l_t(x_j)
                corr = torch.exp(lj - lse.unsqueeze(-1)) * (torch.pow(eps + Pj, beta) - eps ** beta) / cnt.clamp(min=1.0)
                corr = (corr * valid.float()).sum(-1)                                      # sum over distinct context tokens
                logZ = lse + torch.log(eps ** beta + corr)
                py = (w * (idx.unsqueeze(1) == tgt.unsqueeze(2)).float()).sum(-1)
                lpy = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) + beta * torch.log(eps + py) - logZ
                lm = lam.squeeze(-1)
                mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lm + 1e-9) + torch.log(py + 1e-9))
                return -mix.mean()
            if A.cachepoe:   # product of experts: reshape the LM distribution by the cache before (optional) mixing
                pcf = torch.zeros(Bc, Tc, logits.size(-1), device=idx.device).scatter_add_(2, idx.unsqueeze(1).expand(Bc, Tc, Tc), w)
                raw = logits
                logits = logits + F.softplus(s.pbeta) * torch.log(pcf + 1e-3)
                if A.poeonly:
                    if tgt is not None and s.training and A.lmaux > 0:   # fix for the product: plain CE on the LM's own logits
                        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) + A.lmaux * F.cross_entropy(raw.reshape(-1, raw.size(-1)), tgt.reshape(-1))
                    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) if tgt is not None else logits.log_softmax(-1)
            if A.cachefast and tgt is not None and not A.cache2:   # -log((1-l) p_lm(y) + l * sum_j w_tj [x_j == y_t])
                pcy = (w * (idx.unsqueeze(1) == tgt.unsqueeze(2)).float()).sum(-1)
                lpy = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape); lm = lam.squeeze(-1)   # fused, no BxTxV log_softmax
                if A.semhead or A.mos2:   # p_LM' = (1-mu) p_LM + mu p_2; the copy mixture and plain-CE term use p_LM' / p_LM as before
                    E = s.head.weight
                    c = torch.bmm(w.to(E.dtype), E[idx]) if A.semhead else s.mosp(rms(x)).to(E.dtype)
                    l2 = (F.linear(rms(c), E) * s.stau.exp()).float(); l2 = 15 * torch.tanh(l2 / 15)
                    lp2 = -F.cross_entropy(l2.reshape(-1, l2.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)
                    mu = torch.sigmoid(s.cg2(rms(x)).float()).squeeze(-1)
                    lpy = torch.logaddexp(torch.log1p(-mu + 1e-6) + lpy, torch.log(mu + 1e-9) + lp2)
                mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lm + 1e-9) + torch.log(pcy + 1e-9))
                if PRICE[0] is not None and not s.training: PRICE[0].append(torch.stack([lm.detach().float(), lpy.detach().float().exp(), pcy.detach().float()]).cpu())   # Price-equation log: gate, LM prob, copy prob of the target
                if A.cpstyle:   # p = (1 - lam_cp (1 - a)) q + lam_cp * sum over real positions (that sum is already 1 - a): no division
                    mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lam_cp + 1e-9) + torch.log(pcy + 1e-9))
                if A.expert and EXPERT[0] is not None:   # three components: p = (1-l_e) [(1-l) q + l c] + l_e p_expert, expert frozen
                    with torch.no_grad(): le_ = EXPERT[0](idx).float().log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    lme = torch.sigmoid(s.eg(rms(x)).float()).squeeze(-1)
                    mix = torch.logaddexp(torch.log1p(-lme + 1e-6) + mix, torch.log(lme + 1e-9) + le_)
                if s.training and A.hdrop > 0 and torch.rand(()).item() < A.hdrop: return -lpy.mean()   # helper dropout baseline: this micro-batch trains the LM alone
                if A.splitadam and s.training:
                    _T = idx.size(1); SPL[0] = (mix, ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(_T, _T, device=idx.device, dtype=torch.bool))).any(-1).float())
                if CPLOG[0] is not None or A.cpaux > 0:
                    _T = idx.size(1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(_T, _T, device=idx.device, dtype=torch.bool))).any(-1).float()
                    if CPLOG[0] is not None: CPLOG[0].append((torch.log(pcy.float() + 1e-9).cpu(), seen.cpu(), lpy.detach().float().cpu(), mix.detach().float().cpu()))
                if s.training and A.lmaux > 0 and A.cpaux > 0:   # Shapley credit for two components: -(log p + log q + log c), c only where the target is in the context
                    return -mix.mean() - A.lmaux * lpy.mean() - A.cpaux * (torch.log(pcy + 1e-9) * seen).sum() / seen.sum().clamp(min=1.0)
                if s.training and A.lmaux > 0 and DELEG_ON[0]: return -mix.mean() - A.lmaux * lpy.mean()   # keep the LM learning every token (before --deleg_from the no-delegation branch below applies)
                if s.training and (A.nodeleg or not DELEG_ON[0]):   # no delegation: the LM gets plain CE on every token; the mixture trains only cache + gate
                    mixd = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy.detach(), torch.log(lm + 1e-9) + torch.log(pcy + 1e-9))
                    return -mixd.mean() - lpy.mean()
                return -mix.mean()
            pc = torch.zeros(Bc, Tc, logits.size(-1), device=idx.device).scatter_add_(2, idx.unsqueeze(1).expand(Bc, Tc, Tc), w)
            if A.cache2:   # 2-gram context: token[j-1] == token[t] and token[j-2] == token[t-1]
                prev2 = F.pad(idx, (2, 0), value=-2)[:, :-2]; cur1 = F.pad(idx, (1, 0), value=-3)[:, :-1]
                w2 = (idx.unsqueeze(2) == prev.unsqueeze(1)).float() * (cur1.unsqueeze(2) == prev2.unsqueeze(1)).float() * torch.tril(torch.ones(Tc, Tc, device=idx.device))
                has2 = w2.sum(-1, keepdim=True) > 0; w2 = w2 / w2.sum(-1, keepdim=True).clamp(min=1.0)
                pc2 = torch.zeros_like(pc).scatter_add_(2, idx.unsqueeze(1).expand(Bc, Tc, Tc), w2)
                lam2 = torch.sigmoid(gl[..., 1:2]) * has2.float()
                pc = (lam * pc + lam2 * pc2) / (lam + lam2 + 1e-9); lam = 1 - (1 - lam) * (1 - lam2)
            logits = torch.logaddexp(torch.log1p(-lam + 1e-6) + logits.log_softmax(-1), torch.log(lam + 1e-9) + torch.log(pc + 1e-9))
            return F.nll_loss(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) if tgt is not None else logits
        if A.mosk and tgt is not None:   # p = sum_k pi_k softmax(head(tanh(W_k h))); loss -log p (+ optional per-component CE via lmaux); stats in MOS_STATS
            Bm, Tm, dm = x.shape; hk = torch.tanh(s.mosproj(rms(x))).view(Bm, Tm, A.mosk, dm)
            lk = s.head(rms(hk)).float(); lk = 15 * torch.tanh(lk / 15)
            lpk = -F.cross_entropy(lk.reshape(-1, lk.size(-1)), tgt.unsqueeze(-1).expand(-1, -1, A.mosk).reshape(-1), reduction='none').view(Bm, Tm, A.mosk)
            lpi = s.mosgate(rms(x)).float().log_softmax(-1); lmix = torch.logsumexp(lpi + lpk, -1)
            if not s.training:
                resp = (lpi + lpk - lmix.unsqueeze(-1)).exp().mean((0, 1)); MOS_STATS.append((resp.tolist(), (-lpk.mean((0, 1))).tolist()))
            if s.training and A.lmaux > 0: return -lmix.mean() - A.lmaux * lpk.mean()
            return -lmix.mean()
        if A.expert and tgt is not None and EXPERT[0] is not None:   # p = (1-l) p_LM + l p_expert, the expert frozen
            with torch.no_grad():
                le = EXPERT[0](idx).float().log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lpy = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)
            lm = torch.sigmoid(s.eg(rms(x)).float()).squeeze(-1)
            mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lm + 1e-9) + le)
            if s.training and A.lmaux > 0: return -mix.mean() - A.lmaux * LMAUX_SCALE[0] * lpy.mean()   # phased delegation: scale drops to 0 at deleg_at
            return -mix.mean()
        if A.attnptr and tgt is not None and s.training and PTR_ON[0]:   # p = (1-l) q + l c, c from the backbone's pointer-head attention
            lpy = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)
            cy = (PTR[0] * (idx.unsqueeze(1) == tgt.unsqueeze(2)).float()).sum(-1); lm = torch.sigmoid(s.apg(rms(x.detach() if A.gatedetach else x)).float()).squeeze(-1)
            if A.ptrlam >= 0: lm = torch.full_like(lm, A.ptrlam)   # fixed gate: sets the responsibilities directly
            mix = torch.logaddexp(torch.log1p(-lm + 1e-6) + lpy, torch.log(lm + 1e-9) + torch.log(cy + 1e-9))
            _T = idx.size(1); _ind = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(_T, _T, device=idx.device, dtype=torch.bool), -1)).any(-1).float()
            _rc = (lm * cy / ((1 - lm) * lpy.exp() + lm * cy).clamp_min(1e-12)).detach()
            OSTAT[0] = (lm.mean().detach(), cy.mean().detach(), (_rc * _ind).sum() / _ind.sum().clamp(min=1.0), (_rc * (1 - _ind)).sum() / (1 - _ind).sum().clamp(min=1.0))
            return -mix.mean() - (A.lmaux * lpy.mean() if A.lmaux > 0 else 0.0)
        if A.ptraux > 0 and tgt is not None and s.training and PTR_ON[0]:   # baseline: plain CE + separate pointer loss -log c_y on positions whose target is in the context (no mixture)
            cy = (PTR[0] * (idx.unsqueeze(1) == tgt.unsqueeze(2)).float()).sum(-1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(idx.size(1), idx.size(1), device=idx.device, dtype=torch.bool))).any(-1).float()
            if A.ptrrc:   # weight the pointer loss by the responsibility it would have in an equal mixture, r_c = c/(c+q), detached
                qy = torch.exp(-F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)).detach()
                seen = seen * (cy.detach() / (cy.detach() + qy + 1e-9))
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) - A.ptraux * (torch.log(cy + 1e-9) * seen).sum() / seen.sum().clamp(min=1.0)
        if A.indweight > 0 and tgt is not None and s.training:   # plain LM, weight indweight on positions that are NOT induction-copyable, 1 on induction-copyable ones
            Tq = idx.size(1); ind = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool), -1)).any(-1).float()
            nl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape); w = ind + A.indweight * (1 - ind)
            return (nl * w).sum() / w.numel()
        if A.newweight > 0 and tgt is not None and s.training:   # plain LM, loss weight newweight on positions whose target is not yet in the context, 1 on the rest (the fix's 1+r weighting without a copy head)
            Tq = idx.size(1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool))).any(-1).float()
            nl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape); w = seen + A.newweight * (1 - seen)
            return (nl * w).sum() / w.sum()
        if A.maskseen and tgt is not None and s.training:   # Proposition 4 bound: positions whose target already occurs in the context are removed from the loss
            Tq = idx.size(1); seen = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool))).any(-1).float()
            nl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction='none').view(tgt.shape)
            return (nl * (1 - seen)).sum() / (1 - seen).sum().clamp(min=1.0)
        if A.oracle >= 0 and tgt is not None and s.training:   # ungated product with an oracle copy helper on a fraction of copyable positions
            Tq = idx.size(1); seen = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & (tgt.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool), -1)).any(-1)   # induction-copyable: bigram (x_t, y_t) seen before
            m = (seen & (torch.rand(seen.shape, device=idx.device) < (A.oracle if ORC_ON[0] else 0.0))).float(); V = logits.size(-1)
            dl = math.log(((1 - A.oeps) + A.oeps / V) / (A.oeps / V)); lf = logits.float()
            lse = torch.logsumexp(lf, -1); ly = lf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            comb = lse + torch.log1p(torch.exp(ly - lse) * torch.expm1(dl * m)) - ly - dl * m   # -log of the renormalised product at y
            OSTAT[0] = (m.mean().detach(), seen.float().mean().detach())
            if A.falsemask and ORC_ON[0]:   # 1: drop false-copy positions (current token occurred before, bigram not seen) while the oracle is on; 2: drop as many random other positions
                occ = ((idx.unsqueeze(1) == idx.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool), -1)).any(-1); fc = occ & ~seen
                if A.falsemask == 2: rest = ~seen & ~fc; fc = rest & (torch.rand(rest.shape, device=idx.device) < fc.float().sum() / rest.float().sum().clamp(min=1.0))
                if A.falsemask == 3:   # drop new-target positions instead (target not in the context, current token not seen before)
                    sn = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool))).any(-1); fc = ~sn & ~occ & ~seen
                if A.falsemask in (4, 5):   # 4: drop seen-only positions (target in the context, current token new); 5: drop false-copy and new-target positions
                    sn = ((idx.unsqueeze(1) == tgt.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=idx.device, dtype=torch.bool))).any(-1)
                    fc = (sn & ~occ & ~seen) if A.falsemask == 4 else ((occ & ~seen) | (~sn & ~occ & ~seen))
                if A.falsemask == 6: fc = ~seen & (torch.rand(seen.shape, device=idx.device) < A.dropfrac)   # control: drop a random fraction of the non-copyable positions
                if A.falsemask == 7:   # control without context: drop every position whose target is among the 10 most frequent training tokens
                    if FREQTOP[0] is None: FREQTOP[0] = torch.from_numpy(np.load(f"{D}/freq_order.npy")[:10]).to(idx.device)
                    fc = (tgt.unsqueeze(-1) == FREQTOP[0]).any(-1)
                return (comb * (1 - fc.float())).sum() / comb.numel()
            if A.oread > 0 and OREAD_O[0]:   # one prediction from the summed head outputs: the heads share its error
                if A.gate_cfg:   # many independent gate pairs on the same detached outputs: pair k reads g_kA zA + g_kB zB, each gate a product of p_k scalars
                    _sel = seen; _oo = [_o.detach()[_sel][:512].float() for _o in OREAD_O[0]]; OREAD_O[0] = []; _t = tgt[_sel][:512]
                    _W = s.head.weight.detach().float(); _zA = _oo[0] @ _W.t(); _zB = _zA if A.gate_same else _oo[1] @ _W.t(); _ce = 0.0   # gate_same: both gates read the first head (identical cues)
                    for _u, _c in zip(s.gate_us, A.gate_cfg.split(";")): _gv = gate_val(_u, _c); _ce = _ce + F.cross_entropy(_gv[0] * _zA + _gv[1] * _zB, _t) + (float(_c.split(":")[4]) * (_u * (torch.randint(0, 2, _u.shape, device=_u.device) * 2 - 1)).sum() if len(_c.split(":")) > 4 else 0.0) + (float(_c.split(":")[5]) * _u.sum() if len(_c.split(":")) > 5 else 0.0)   # 5th field: random-sign gradient of this size on every factor; 6th: constant pull of this size toward zero (a consistent competing objective)
                    return comb.mean() + A.oread * _ce
                if A.gate_p > 0:   # gated readout of the detached head outputs: each gate is a product of gate_p scalars (only the gates learn here)
                    _g = s.gate_u.prod(1); _os = sum(_g[_k] * _o.detach().float() for _k, _o in enumerate(OREAD_O[0])); _sel = seen; OREAD_O[0] = []
                    _ce = F.cross_entropy(F.linear(_os[_sel], s.head.weight.detach().float()), tgt[_sel]) if _sel.any() else 0.0
                    return comb.mean() + A.oread * _ce
                _os = sum(OREAD_O[0]); _sel = seen
                _ce = A.oread_rate * A.oread_T * F.cross_entropy(s.head(rms(_os[_sel])).float() / A.oread_T, tgt[_sel]) if _sel.any() else 0.0   # oread_T > 1: outcome upshift (the summed heads fall short, larger shared error at the same rate); < 1: downshift
                if A.oread_own > 0 and _sel.any():   # Bush-Mosteller / Shapley: each head also predicts the target from its own output (its own error)
                    _ce = _ce + A.oread_own * sum(F.cross_entropy(s.head(rms(_o[_sel])).float(), tgt[_sel]) for _o in OREAD_O[0])
                OREAD_O[0] = []
                return comb.mean() + A.oread * _ce
            if A.nudge > 0 and NUDGE_ON[0] and NUDGE_P[0]:   # -log of each pulled head's attention on successors of earlier occurrences of the current token
                _ps = F.pad((idx.unsqueeze(1) == idx.unsqueeze(2)).float(), (1, 0))[:, :, :Tq] * torch.tril(torch.ones(Tq, Tq, device=idx.device), 0); _has = _ps.sum(-1) > 0
                if A.nudge_prev: _ps = torch.diag_embed(torch.ones(Tq - A.nudge_off, device=idx.device), -A.nudge_off).expand(idx.size(0), -1, -1); _has = _ps.sum(-1) > 0   # pull toward the previous position instead (a non-induction pattern)
                _aux = sum((-torch.log((_p * _ps).sum(-1) + 1e-6) * _has).sum() / _has.sum().clamp(min=1.0) for _p in NUDGE_P[0]); NUDGE_P[0] = []
                return comb.mean() + A.nudge * _aux
            return comb.mean() + (A.lmaux * (lse - ly).mean() if A.lmaux > 0 else 0.0)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) if tgt is not None else logits


EXPERT = [None]
PRICE = [None]
LMAUX_SCALE = [1.0]
MOS_STATS = []
D = "./data"
tr = np.memmap(f"{D}/{A.train_file}", dtype=np.uint16, mode="r"); va = np.memmap(f"{D}/val.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(A.seed)
def batch(arr, bs, r):
    ix = r.integers(0, len(arr) - A.seq - 1, bs)
    x = torch.from_numpy(np.stack([arr[i:i + A.seq + 1].astype(np.int64) for i in ix]))
    ATT_MASK[0] = None
    if A.facts and (arr is not va or os.environ.get('FACTS_VAL')): x = facts_inject(x, r)
    if A.mdlm:   # masked diffusion: per-sequence mask rate t ~ U(tmin, tmax); targets only at masked positions (token 50257 is the mask)
        x = x[:, :-1]; t = torch.from_numpy(r.uniform(A.tmin, A.tmax, (x.size(0), 1))).float()
        mk = torch.rand(x.shape) < t; inp = x.masked_fill(mk, 50257); tg = x.masked_fill(~mk, -100)
        return inp.to(dev, non_blocking=True), tg.to(dev, non_blocking=True)
    return x[:, :-1].to(dev, non_blocking=True), x[:, 1:].to(dev, non_blocking=True)
_FJ = json.load(open(A.facts)) if A.facts else {}
FUNITS = _FJ.get("units") or [dict(docs=[d], mask=False) for ds in _FJ.get("docs", []) for d in ds]   # a unit = documents pasted into one sequence
def facts_inject(x, r):   # --facts: with probability FACTP each sequence gets one fact unit; one document at a random offset, two in the first and second half
    p = float(os.environ.get("FACTP", "0.1")); T = x.size(1) - 1; M = None; ATT_MASK[0] = None
    for b in range(x.size(0)):
        if r.random() < p:
            u = FUNITS[r.integers(len(FUNITS))]; ds = u["docs"]
            if len(ds) == 1: d = ds[0]; o = r.integers(0, x.size(1) - len(d)); x[b, o:o + len(d)] = torch.tensor(d); continue
            o1 = r.integers(0, T // 2 - len(ds[0])); o2 = r.integers(T // 2, T - len(ds[1]))
            x[b, o1:o1 + len(ds[0])] = torch.tensor(ds[0]); x[b, o2:o2 + len(ds[1])] = torch.tensor(ds[1])
            if u["mask"]:
                if M is None: M = torch.ones(T, T, dtype=torch.bool).tril().expand(x.size(0), 1, T, T).clone()
                M[b, 0, o2:o2 + len(ds[1]), o1:o1 + len(ds[0])] = False
    if M is not None: ATT_MASK[0] = M.to(dev, non_blocking=True)
    return x

model = GPT().to(dev); nparam = sum(p.numel() for p in model.parameters())
if A.init:   # continue from a saved checkpoint; modules it lacks (e.g. a new copy head) keep their fresh init
    _sd0 = torch.load(A.init, map_location=dev); _mk = model.load_state_dict({_a: _b.float() if _b.is_floating_point() else _b for _a, _b in _sd0.items()}, strict=False)
    print("INIT", A.init, "missing", len(_mk.missing_keys), "unexpected", len(_mk.unexpected_keys), flush=True)
if A.expert:   # frozen external expert: rebuilt from its own config in a separate namespace
    import sys
    import json as _js
    _ecfg = _js.load(open(A.expert))["cfg"]; _argv = ["x"]
    for _k, _v in _ecfg.items():
        if _k in ("out", "expert"): continue
        if isinstance(_v, bool):
            if _v: _argv.append("--" + _k)
        else: _argv += ["--" + _k, str(_v)]
    _src = open(__file__).read().split("EXPERT = [None]\n")[0]; _old = sys.argv; sys.argv = _argv; _g = {"__name__": "expert"}
    try: exec(_src, _g)
    finally: sys.argv = _old
    _em = _g["GPT"]().to(dev); _sd = torch.load(A.expert.replace(".json", ".pt"), map_location=dev)
    _em.load_state_dict({_a: _b.float() if _b.is_floating_point() else _b for _a, _b in _sd.items()}); _em.eval()
    for _p in _em.parameters(): _p.requires_grad_(False)
    EXPERT[0] = _em; print("EXPERT loaded", A.expert, sum(_p.numel() for _p in _em.parameters()), flush=True)
if A.freeze_base:   # two-stage baseline: only the copy head and its gate learn on top of a frozen, separately trained LM
    for _n, _p in model.named_parameters(): _p.requires_grad_(_n.split(".")[0] in ("cth", "cmix", "cq", "ck", "cg"))
    print("trainable", [n for n, p in model.named_parameters() if p.requires_grad], flush=True)
def _ns(G, steps=5):   # Newton-Schulz orthogonalisation (Muon)
    a, b, c = 3.4445, -4.7750, 2.0315; X = G.float() / (G.norm() + 1e-7); tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps): Am = X @ X.T; X = a * X + (b * Am + c * Am @ Am) @ X
    return X.T if tr else X
class Muon(torch.optim.Optimizer):   # --muon LR: momentum 0.95 (Nesterov), orthogonalised update per weight matrix of the blocks
    def __init__(s, ps, lr=0.02, mom=0.95): super().__init__(ps, dict(lr=lr, mom=mom))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                buf = s.state[p].setdefault("m", torch.zeros_like(p)); buf.mul_(g["mom"]).add_(p.grad)
                u = _ns(p.grad.add(buf, alpha=g["mom"])); p.add_(u.to(p.dtype), alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)
MUON_OPT = [None]
_mu = [p for n, p in model.named_parameters() if A.muon > 0 and p.requires_grad and p.dim() == 2 and n.startswith("blocks.")]
if _mu: MUON_OPT[0] = Muon(_mu, lr=A.muon); print("muon on", len(_mu), "matrices", flush=True)
decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(p is q for q in _mu) and p.dim() >= 2 and "emb" not in n]
nodecay = [p for n, p in model.named_parameters() if p.requires_grad and not any(p is q for q in _mu) and not (p.dim() >= 2 and "emb" not in n)]
GOPT = [None]; GOPT2 = [None]
def gate_val(u, c):   # 7th field M > 0: bounded gate M * sigmoid(prod(u) - 5), else the plain product
    f = c.split(":"); M = float(f[6]) if len(f) > 6 else 0.0
    return torch.stack([M * torch.sigmoid(u[k].prod() - 5.0) if M > 0 else u[k].prod() for k in range(2)])
if A.gate_cfg:   # "p:gA:gB:s|a;..." s = SGD (gate_lr), a = Adam (lr 0.003)
    _cf = [c.split(":") for c in A.gate_cfg.split(";")]
    def _ginit(c, g):   # factor value giving gate g: plain product g^(1/p); bounded: (logit(g/M) + 5)^(1/p)
        M = float(c[6]) if len(c) > 6 else 0.0; z = float(g) if M <= 0 else math.log(float(g) / M / (1 - float(g) / M)) + 5.0
        return z ** (1.0 / int(c[0]))
    model.gate_us = nn.ParameterList([nn.Parameter(torch.tensor([[_ginit(c, c[1])] * int(c[0]), [_ginit(c, c[2])] * int(c[0])], device=dev)) for c in _cf])
    _sg = [u for u, c in zip(model.gate_us, _cf) if c[3] == "s"]; _ad = [u for u, c in zip(model.gate_us, _cf) if c[3] == "a"]
    GOPT[0] = torch.optim.SGD(_sg, lr=A.gate_lr) if _sg else None; GOPT2[0] = torch.optim.Adam(_ad, lr=0.003) if _ad else None
if A.gate_p > 0:   # gates registered after the optimiser groups are formed, with their own optimiser (SGD unless --gate_adam)
    model.gate_u = nn.Parameter(torch.tensor([[float(_v) ** (1.0 / A.gate_p)] * A.gate_p for _v in A.gate_g0.split(",")], device=dev))
    GOPT[0] = torch.optim.Adam([model.gate_u], lr=A.gate_lr) if A.gate_adam else torch.optim.SGD([model.gate_u], lr=A.gate_lr)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": A.wd}, {"params": nodecay, "weight_decay": 0.0}], lr=A.lr, betas=(0.9, 0.95), fused=True)
steps = int(A.tokens // (A.bs * A.seq)); warm = max(1, steps // 50)
def lr_at(i): return A.lr * (i + 1) / warm if i < warm else A.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
cmodel = torch.compile(model)

@torch.no_grad()
def evaluate():
    SOFT_OFF[0] = False; WIN[0] = 0; model.eval(); r = np.random.default_rng(1234); n = int(A.eval_tokens // (A.micro * A.seq)); tot = 0.0
    for _ in range(n):
        x, y = batch(va, A.micro, r)
        with torch.autocast("cuda", torch.bfloat16): tot += cmodel(x, y).item()
    model.train(); return tot / n

_ir = torch.Generator(device="cpu").manual_seed(4242); _half = torch.randint(0, 5000, (8, min(256, A.seq // 2)), generator=_ir); IND_X = torch.cat([_half, _half], 1).cuda()
def induction():   # LM-alone loss on the second copy of a repeated random sequence (standard induction score)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        lp = model(IND_X[:, :-1]).float().log_softmax(-1).gather(-1, IND_X[:, 1:].unsqueeze(-1)).squeeze(-1)
    model.train(); h = IND_X.size(1) // 2; return (lp[:, :h - 1].mean() - lp[:, h:].mean()).item()   # 2nd-half minus 1st-half loss (<0: induction)
def headind():   # induction attention (t -> token after the previous occurrence) of the best head, and of head 0 in the last layer
    model.eval(); ATTN[0] = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): model(IND_X)
    h = IND_X.size(1) // 2; t = torch.arange(h + 1, 2 * h); sc = [a[:, :, t, t - h + 1].mean((0, 2)) for a in ATTN[0]]; ATTN[0] = None; model.train()
    return round(max(x.max().item() for x in sc), 4), round(sc[A.ptrlayer if A.ptrlayer >= 0 else -1][0].item(), 4)
_cr = np.random.default_rng(999); COPY_B = [batch(va, A.micro, _cr) for _ in range(4)]
def copysplit():   # LM-alone val loss on induction-copyable positions (bigram (x_t, y_t) seen before in the sequence) vs the rest
    model.eval(); a = b = na = nb = 0.0
    with torch.no_grad():
        for x, y in COPY_B:
            with torch.autocast("cuda", torch.bfloat16): lg = model(x)
            nll = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="none").view(y.shape); Tq = x.size(1)
            m = ((x.unsqueeze(1) == x.unsqueeze(2)) & (y.unsqueeze(1) == y.unsqueeze(2)) & torch.tril(torch.ones(Tq, Tq, device=x.device, dtype=torch.bool), -1)).any(-1).float()
            a += (nll * m).sum().item(); na += m.sum().item(); b += (nll * (1 - m)).sum().item(); nb += (1 - m).sum().item()
    model.train(); return round(a / na, 4), round(b / nb, 4)
OSTAT = [None]
os.makedirs(A.out, exist_ok=True); log = []; t0 = time.time(); PLOG = []
print(f"{A.name} params {nparam/1e6:.1f}M steps {steps}", flush=True)
for i in range(steps):
    for g in opt.param_groups: g["lr"] = lr_at(i)
    SOFT_OFF[0] = A.soft_drop > 0 and float(torch.rand(())) < A.soft_drop
    WIN[0] = A.win if A.winp > 0 and float(torch.rand(())) < A.winp else 0
    if MUON_OPT[0] is not None:
        for g in MUON_OPT[0].param_groups: g["lr"] = A.muon * lr_at(i) / A.lr
    if A.deleg_at >= 0: LMAUX_SCALE[0] = 0.0 if i >= A.deleg_at * steps else 1.0
    PTR_ON[0] = i < A.ptr_until * steps; ABL_ON[0] = bool(A.ablate) and int(A.ablate.split(',')[2]) <= i < (int(A.ablate.split(',')[3]) if len(A.ablate.split(',')) > 3 else 10**9); OREAD_ACT[0] = [(int(a), int(b)) for a, b, c, e in (w.split(',') for w in A.oread_spec.split(';')) if int(c) <= i < int(e)] if A.oread_spec else []; SOLO_ACT[0] = [(int(a), int(b)) for a, b, c, e in (w.split(',') for w in A.solo_spec.split(';')) if int(c) <= i < int(e)] if A.solo_spec else []; OREAD_O[0] = []; NUDGE_ACT[0] = ([(int(a), int(b)) for a, b, c, e in (w.split(',') for w in A.nudge_spec.split(';')) if int(c) <= i < int(e)] if A.nudge_spec else ([tuple(int(v) for v in A.nudge_lh.split(','))] if i < A.nudge_until else [])); NUDGE_ON[0] = len(NUDGE_ACT[0]) > 0; NUDGE_P[0] = []; DELEG_ON[0] = (i >= A.deleg_from) if A.autodeleg <= 0 else AUTO[0]; ORC_ON[0] = (A.orc_from <= i < A.orc_until) if not A.orc_windows else any(int(w.split(':')[0]) <= i < int(w.split(':')[1]) for w in A.orc_windows.split(','))
    acc = A.bs // A.micro
    if A.splitadam:   # responsibility-split Adam: gradients from seen-target and new-target positions get separate moment estimates
        PL = [p for p in model.parameters() if p.requires_grad]; GB = [torch.zeros_like(p) for p in PL]; GU = [torch.zeros_like(p) for p in PL]
        for _ in range(acc):
            x, y = batch(tr, A.micro, rng)
            with torch.autocast("cuda", torch.bfloat16): loss = cmodel(x, y)
            mx, sn = SPL[0]; n = mx.numel() * acc
            if A.splitrand: sn = (torch.rand_like(sn) < sn.mean()).float()   # control: same split sizes, positions chosen at random
            for G, l, keep in ((GB, -(mx * sn).sum() / n, True), (GU, -(mx * (1 - sn)).sum() / n, False)):
                gs = torch.autograd.grad(l, PL, retain_graph=keep, allow_unused=True)
                for a_, g_ in zip(G, gs):
                    if g_ is not None: a_.add_(g_)
        if i == 0: SA = {id(p): [torch.zeros_like(p) for _ in range(4)] for p in PL}; WD = {id(p): gr["weight_decay"] for gr in opt.param_groups for p in gr["params"]}
        b1, b2, lr_, t1 = 0.9, 0.95, lr_at(i), i + 1
        with torch.no_grad():
            for p, gb_, gu_ in zip(PL, GB, GU):
                st_ = SA[id(p)]; p.mul_(1 - lr_ * WD.get(id(p), 0.0)); upd = torch.zeros_like(p)
                for g_, m_, v_ in ((gb_, st_[0], st_[1]), (gu_, st_[2], st_[3])):
                    m_.mul_(b1).add_(g_, alpha=1 - b1); v_.mul_(b2).addcmul_(g_, g_, value=1 - b2)
                    upd += (m_ / (1 - b1 ** t1)) / ((v_ / (1 - b2 ** t1)).sqrt() + 1e-8)
                p.sub_(lr_ * upd)
    else:
        for _ in range(acc):
            x, y = batch(tr, A.micro, rng)
            with torch.autocast("cuda", torch.bfloat16): loss = cmodel(x, y)
            (loss / acc).backward(); ATT_MASK[0] = None
        (GOPT[0].step(), GOPT[0].zero_grad(set_to_none=True)) if GOPT[0] is not None else None
        (GOPT2[0].step(), GOPT2[0].zero_grad(set_to_none=True)) if GOPT2[0] is not None else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if MUON_OPT[0] is not None: MUON_OPT[0].step(); MUON_OPT[0].zero_grad(set_to_none=True)
    if A.pricelog and (i % 250 == 0 or i == steps - 1):   # per-token (gate, q, c) on a fixed batch, for the Price decomposition
        with torch.no_grad():
            model.eval(); PRICE[0] = []; _r = np.random.default_rng(777); _x, _y = batch(va, 8, _r)
            with torch.autocast('cuda', torch.bfloat16): model(_x, _y)
            PLOG.append((i, PRICE[0][0].numpy())); PRICE[0] = None; model.train()
    if A.save_at and str(i) in A.save_at.split(","): torch.save({_k: _v for _k, _v in model.state_dict().items() if not _k.startswith("gate_u")}, f"{A.out}/{A.name}_s{A.seed}_step{i}.pt")
    if A.autodeleg > 0 and not AUTO[0] and i % 50 == 0:
        _h = headind()[0]
        if _h > A.autodeleg: AUTO[0] = True; print(json.dumps(dict(autodeleg_step=i, head=_h)), flush=True)
    if i % 200 == 0 or i == steps - 1 or (i < 1000 and i % 50 == 0):
        rec = dict(step=i, loss=loss.item(), t=time.time() - t0, ind=induction(), cs=copysplit(), hi=headind())
        if A.gate_cfg: rec["gates"] = [[round(float(v), 5) for v in gate_val(u.detach(), c)] for u, c in zip(model.gate_us, A.gate_cfg.split(";"))]
        if A.gate_p > 0: rec["gate"] = [round(float(v), 5) for v in model.gate_u.detach().prod(1)]
        if OSTAT[0] is not None: rec["cov"] = [round(float(v), 4) for v in OSTAT[0]]
        if i % 1000 == 0 or i == steps - 1: rec["val"] = evaluate()
        log.append(rec); print(json.dumps(rec), flush=True)
res = dict(name=A.name, seed=A.seed, cfg=vars(A), params=nparam, val=log[-1]["val"], secs=time.time() - t0, log=log)
if A.mosk and MOS_STATS:
    import numpy as _np
    _r = _np.mean([a for a, b in MOS_STATS[-50:]], 0); _l = _np.mean([b for a, b in MOS_STATS[-50:]], 0)
    print("MOS responsibility", _np.round(_r, 3).tolist(), "component-alone loss", _np.round(_l, 3).tolist(), flush=True); res["mos"] = dict(resp=_r.tolist(), comp_loss=_l.tolist())
json.dump(res, open(f"{A.out}/{A.name}_s{A.seed}.json", "w"))
if PLOG: np.savez(f"{A.out}/{A.name}_s{A.seed}_price.npz", steps=np.array([a for a, b in PLOG]), g=np.stack([b for a, b in PLOG]))
if not A.nosave: torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}, f"{A.out}/{A.name}_s{A.seed}.pt")
print("FINAL", A.name, A.seed, res["val"], f"{res['secs']:.0f}s", flush=True)
