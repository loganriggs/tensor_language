"""FAN-OUT B (dossier attn-middle-pooling / benchmark: the middle collective pool has NO stand-in yet).
§1085: middle patterns are POSITIONAL/recency-first (log-dist r -0.39..-0.61). If that is the function (not just
the correlate), a STATIC POSITIONAL KERNEL should reproduce the middle pool: replace each middle head's pattern
with k_h(d) — its own mean pattern value as a function of key distance d (fit from data, position-only) — while
keeping VALUES dynamic. Compare per middle band L6-14: (i) heads fully const (bias only; §1093 floor for the
band), (ii) kernel-pattern (static routing, dynamic values), (iii) base. Kernel recovery = 1 - kernel_cost /
const_cost. NSEQ=192.

REGISTERED PREDICTIONS:
  (0) SANITY: kernel cost <= const cost (kernel keeps strictly more structure); base ~0.
  (a) KERNEL STAND-IN WORKS: the positional kernel recovers >= 0.6 of the middle band's collective dynamic value
      -> the middle pool is understood as 'fixed recency kernel over dynamic values' — a benchmark win for the
      band (the §1085 correlational picture is the causal function);
  (b) if kernel recovery < 0.3, the pattern's DEVIATION from positional (the content-sim second-order part,
      §1085) is the load-bearing piece — the correlational ranking inverted causally (report plainly; then the
      §1085 'second-order' framing is wrong for the collective function);
  (c) report the same for the GATHERER band L3-5 — §1074/§fan-out-A implicate it in the content seed; if the
      kernel works for L6-14 but NOT L3-5, the seed-gathering is the one genuinely content-adaptive routing."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_kernel_standin_results.json'
NSEQ = 192; SEQ = 256
BANDS = {'gatherer_L3_5': [3, 4, 5], 'middle_L6_14': list(range(6, 15))}
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CTL = {'mode': None, 'band': None}   # mode: 'const' | 'kernel'
MEANS = {}; KERNELS = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def const_hook(L):
    def h(mo, args):
        if CTL['mode'] != 'const' or L not in CTL['band']: return None
        y = args[0].clone()
        for hh in range(NH):
            y[..., hh*HD:(hh+1)*HD] = MEANS[L][hh].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


def make_sq(attn, L):
    orig = attn.squared_attention
    def patched(self, q, k, v, q2, k2):
        B, T, Hh, Dh = q.shape
        if CTL['mode'] == 'kernel' and L in CTL['band']:
            pat = KERNELS[L][:, :T, :T].unsqueeze(0).expand(B, NH, T, T).to(v.dtype)
        else:
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
            pat = (s1/Dh)*(s2/Dh)
            cm = torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool))
            pat = pat.masked_fill(cm.logical_not(), 0.0)
        return torch.einsum('bhqk,bkhd->bhqd', pat, v)
    return orig, types.MethodType(patched, attn)


@torch.no_grad()
def pattern_for(attn, x):
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    ALL_L = sorted({L for ls in BANDS.values() for L in ls})
    T = SEQ - 1

    # pass 1: head means + per-head distance kernels (mean pattern value at each distance)
    CAPX = {}
    hcap = [H[L].attn.register_forward_pre_hook((lambda L: (lambda mo, args: CAPX.__setitem__(L, args[0].detach())))(L)) for L in ALL_L]
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in ALL_L}
    hmean = []
    for L in ALL_L:
        def mk(L):
            def h(mo, args): caps[L] += args[0].detach().float().reshape(-1, NH, HD).sum(0)
            return h
        hmean.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    ksum = {L: torch.zeros(NH, T, device=DEV) for L in ALL_L}
    kcnt = torch.zeros(T, device=DEV)
    npos = 0; nb = 0
    di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)
    for i in range(0, 64, 8):    # 64 seqs enough for kernel fit
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); npos += idx.numel(); nb += 1
        for L in ALL_L:
            pat = pattern_for(H[L].attn, CAPX[L])   # B,NH,T,T
            for dd in range(T):
                mask = (di == dd)
                nel = int(mask.sum())
                if nel == 0: continue
                ksum[L][:, dd] += pat[:, :, mask].sum((0, 2))
                if L == ALL_L[0]: kcnt[dd] += nel * pat.shape[0]
            del pat
    for h in hmean + hcap: h.remove()
    # finish means with remaining data (means from 64 seqs are fine; scale)
    for L in ALL_L: MEANS[L] = caps[L] / npos
    for L in ALL_L:
        kern = ksum[L] / kcnt.clamp_min(1)          # NH, T (mean pattern at distance d)
        km = torch.zeros(NH, T, T, device=DEV)
        for dd in range(T):
            vals = kern[:, dd]
            iidx = torch.arange(dd, T, device=DEV)
            km[:, iidx, iidx-dd] = vals.unsqueeze(1)
        KERNELS[L] = km

    origs = {}
    for L in ALL_L:
        o, p = make_sq(H[L].attn, L); origs[L] = o; H[L].attn.squared_attention = p
    hconst = [H[L].attn.c_proj.register_forward_pre_hook(const_hook(L)) for L in ALL_L]
    CTL['mode'] = None; base = ce(blocks)
    out = {'base_ce': round(base, 4), 'bands': {}}
    for bname, ls in BANDS.items():
        row = {}
        for mode in ['const', 'kernel']:
            CTL['mode'] = mode; CTL['band'] = set(ls)
            row[mode] = round(ce(blocks) - base, 4)
            CTL['mode'] = None; CTL['band'] = None
        row['kernel_recovery'] = round(1 - row['kernel']/max(row['const'], 1e-6), 3)
        out['bands'][bname] = row
        print(f"{bname}: const +{row['const']} | kernel +{row['kernel']} | kernel recovery {row['kernel_recovery']}", flush=True)
    for L in ALL_L: H[L].attn.squared_attention = origs[L]
    for h in hconst: h.remove()

    km = out['bands']['middle_L6_14']['kernel_recovery']; kg = out['bands']['gatherer_L3_5']['kernel_recovery']
    out['pred_a_kernel_works_middle'] = bool(km >= 0.6)
    out['pred_b_content_routing_loadbearing'] = bool(km < 0.3)
    out['pred_c_gatherer_content_adaptive'] = bool(km >= 0.6 and kg < 0.4)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a kernel-works(middle) {out['pred_a_kernel_works_middle']} | pred_b content-routing {out['pred_b_content_routing_loadbearing']} | pred_c gatherer-adaptive {out['pred_c_gatherer_content_adaptive']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
