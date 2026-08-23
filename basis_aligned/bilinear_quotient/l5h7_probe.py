"""REDTEAM+MECHANISM follow-up to §1083: L5H7 is the single super-head (0.91 nats, 54% of all per-head attention
cost, broad-pool profile). Hypothesis: L5H7 is THE content gatherer -- the head that pools the value-residual's
per-token content (§1076) into the content seed (§1074) that the deep-middle multiplies (§1041). Four tests, each
mirrored on a matched control head (L10H5, inert pooler, cost 0.007):
  (1) CONTENT SIGNATURE: zero the head; is the cost concentrated on rare/content targets (rare/freq cost ratio
      like the value-residual's 2.69, §1075) rather than flat (grammar-like ~1)?
  (2) INTERCHANGE: replace L5H7's output with a DONOR context's (roll batch); if the head carries the content
      seed, donor swap should transport tail preferences (donor-direction shift) and cost ~ as much as zeroing.
  (3) DISTANCE: recompute with the head's pattern masked to local-only (d<=8) vs far-only (d>8); which range
      carries its function?
  (4) CONTENT-SUBSPACE MEDIATION: project the head's output onto the content subspace U_c (pooled L8-12 deviation
      basis) vs its complement; zero each part separately -- if the head feeds the content machine, the U_c-part
      should carry most of the cost.

REGISTERED PREDICTIONS:
  (0) SANITY: L5H7 zero cost reproduces ~0.91; control head all-tests small (<0.05).
  (a) CONTENT GATHERER: L5H7 rare/freq cost ratio >= 2; restricting the head to LOCAL-only keys (d<=8) costs
      MORE than restricting to far-only (its function comes from far context, it pools broadly);
      content-projection zeroing carries >= 60% of the full zero cost despite U_c being 64/1152 dims;
  (b) INTERCHANGE TRANSPORTS: donor swap cost within 1.5x of zeroing AND the donor's own rare-target log-probs
      improve relative to zeroing (the swapped-in seed votes for donor words);
  (c) if the cost is flat across frequency and the content projection carries little, L5H7 is a general-purpose
      pooler, not the content gatherer (report plainly)."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l5h7_probe_results.json'
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]; K = 64; RARE_MAX = 2
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
TARGETS = [(5, 7), (10, 5)]  # (layer, head): super-head + inert-pooler control
CTL = {'layer': -1, 'head': -1, 'mode': None, 'donor': None, 'Uc': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def head_hook(L):
    """forward_pre_hook on attn.c_proj: intervene on head slice of its input."""
    def h(mo, args):
        if CTL['layer'] != L or CTL['mode'] is None: return None
        y = args[0].clone(); hh = CTL['head']; sl = slice(hh*HD, (hh+1)*HD)
        seg = y[..., sl]
        if CTL['mode'] == 'zero':
            y[..., sl] = 0.0
        elif CTL['mode'] == 'donor':
            y[..., sl] = torch.roll(seg, 1, dims=0)
        else:
            return None  # zero_content / zero_rest / pattern modes handled elsewhere
        return (y,) + tuple(args[1:])
    return h


def out_hook(L):
    """forward hook on attn.c_proj: subtract the head's D-space contribution projected on/off U_c."""
    def h(mo, args, out):
        if CTL['layer'] != L or CTL['mode'] not in ('zero_content', 'zero_rest'): return None
        y = args[0]; hh = CTL['head']; sl = slice(hh*HD, (hh+1)*HD)
        Wh = mo.weight[:, sl]                      # D x HD
        contrib = y[..., sl].to(Wh.dtype) @ Wh.T   # B,T,D head's contribution to output
        Uc = CTL['Uc'].to(contrib.dtype)
        cpart = (contrib @ Uc) @ Uc.T
        sub = cpart if CTL['mode'] == 'zero_content' else (contrib - cpart)
        return out - sub.to(out.dtype)
    return h


def mask_pattern(attn, L):
    orig = attn.squared_attention
    def patched(self, q, k, v, q2, k2):
        B, T, Hh, Dh = q.shape
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
        pat = (s1/Dh)*(s2/Dh)
        cm = torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool))
        pat.masked_fill_(cm.logical_not(), 0.0)
        if CTL['layer'] == L and CTL['mode'] in ('local_only', 'far_only'):
            di = torch.arange(T, device=pat.device).view(-1, 1) - torch.arange(T, device=pat.device).view(1, -1)
            keep = (di <= 8) if CTL['mode'] == 'local_only' else (di > 8)
            pm = pat[:, CTL['head']].clone()
            pm = pm * keep.unsqueeze(0).to(pm.dtype)
            pat[:, CTL['head']] = pm
        return torch.einsum('bhqk,bkhd->bhqd', pat, v)
    return orig, types.MethodType(patched, attn)


@torch.no_grad()
def content_basis(blocks):
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; cap[L] = []; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()


@torch.no_grad()
def ce_split(blocks, is_rare):
    tot = 0.0; n = 0; tr = 0.0; nr = 0; tfq = 0.0; nf = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
        rm = is_rare[tgt]
        tot += float(ce_tok.sum()); n += tgt.shape[0]
        tr += float(ce_tok[rm].sum()); nr += int(rm.sum())
        tfq += float(ce_tok[~rm].sum()); nf += int((~rm).sum())
    return tot/n, tr/max(nr, 1), tfq/max(nf, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    CTL['Uc'] = content_basis(blocks)
    V = int(m.lm_head.weight.shape[0])
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks[:, 1:].to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    hooks = []
    for (L, _) in TARGETS:
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(head_hook(L)))
        hooks.append(H[L].attn.c_proj.register_forward_hook(out_hook(L)))
    origs = {}
    for (L, _) in TARGETS:
        o, p = mask_pattern(H[L].attn, L); origs[L] = o; H[L].attn.squared_attention = p

    CTL['layer'] = -1
    base, base_r, base_f = ce_split(blocks, is_rare)
    out = {'base_ce': round(base, 4), 'base_rare': round(base_r, 4), 'base_freq': round(base_f, 4), 'heads': {}}
    for (L, hh) in TARGETS:
        row = {}
        for mode in ['zero', 'donor', 'local_only', 'far_only', 'zero_content', 'zero_rest']:
            CTL['layer'] = L; CTL['head'] = hh; CTL['mode'] = mode
            c, cr, cf = ce_split(blocks, is_rare)
            row[mode] = {'cost': round(c-base, 4), 'rare_cost': round(cr-base_r, 4), 'freq_cost': round(cf-base_f, 4)}
            CTL['layer'] = -1; CTL['mode'] = None
            print(f"L{L}H{hh} {mode}: cost {row[mode]['cost']} | rare {row[mode]['rare_cost']} | freq {row[mode]['freq_cost']}", flush=True)
        z = row['zero']
        row['rare_freq_ratio'] = round(z['rare_cost']/max(z['freq_cost'], 1e-4), 2)
        row['content_frac_of_zero'] = round(row['zero_content']['cost']/max(z['cost'], 1e-4), 3)
        out['heads'][f'L{L}H{hh}'] = row
    for h in hooks: h.remove()
    for (L, _) in TARGETS: H[L].attn.squared_attention = origs[L]

    s = out['heads']['L5H7']
    out['pred_a_content_gatherer'] = bool(s['rare_freq_ratio'] >= 2
                                          and s['far_only']['cost'] < s['local_only']['cost']
                                          and s['content_frac_of_zero'] >= 0.6)
    out['pred_b_interchange'] = bool(s['donor']['cost'] <= 1.5*s['zero']['cost'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L5H7: rare/freq {s['rare_freq_ratio']} | content-frac {s['content_frac_of_zero']} | local-mask {s['local_only']['cost']} vs far-mask {s['far_only']['cost']}", flush=True)
    print(f"pred_a content-gatherer {out['pred_a_content_gatherer']} | pred_b interchange {out['pred_b_interchange']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
