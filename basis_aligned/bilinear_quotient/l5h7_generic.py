"""Follow-up to §1087 (registered there): L5H7's zeroing costs 0.88 nats but a same-position donor swap costs only
0.02 — the load-bearing signal looks DOCUMENT-GENERIC. Clean decomposition of the head's function into GENERIC
(position-dependent, document-independent) vs SPECIFIC (document-dependent) parts, plus the ramp hypothesis:
  (1) BATCH-MEAN replacement: replace L5H7's output at each position t with the batch-mean output at t
      (keeps the generic positional component exactly, deletes ALL document-specificity);
  (2) POSITION-SHUFFLED donor: replace position t's output with the same DOCUMENT's output at a random t'
      (keeps document content, breaks the positional component);
  (3) NORM-ONLY: replace with a vector of the same per-position norm but batch-mean direction... covered by (1);
      instead: GLOBAL-mean replacement (one vector everywhere -- deletes the positional ramp too; the delta
      between global-mean and batch-mean replacement isolates the ramp's value);
  (4) RAMP CHECK (measurement): the head's output norm vs position curve (is there a strong ramp? unnormalized
      squared attention predicts output mass grows with context length).

REGISTERED PREDICTIONS:
  (0) SANITY: zero cost reproduces ~0.88; batch-mean <= donor cost (~0.02, §1087) since donor = one sample of the
      distribution batch-mean averages over.
  (a) GENERIC-POSITIONAL FUNCTION: batch-mean replacement is near-free (< 0.05) AND global-mean replacement is
      much costlier (> 0.3) -> the head's causal value = the position-dependent generic component (the ramp),
      confirming §1087's reframe; output norm rises markedly with position (final/early ratio > 1.5);
  (b) SPECIFIC CONTENT SECONDARY: position-shuffled same-doc donor costs MORE than batch-mean (position matters
      more than document identity at this node);
  (c) if batch-mean is costly (> 0.3), the specific content DOES matter and §1087's donor result was a fluke of
      adjacent-document similarity -- report plainly."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'l5h7_generic_results.json'
NSEQ = 96; SEQ = 256; L = 5; HH = 7; RARE_MAX = 2
H = m.transformer.h
CTL = {'mode': None, 'mean_t': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def head_hook(mo, args):
    if CTL['mode'] is None: return None
    y = args[0].clone(); sl = slice(HH*HD, (HH+1)*HD)
    seg = y[..., sl]; B, T, _ = seg.shape
    if CTL['mode'] == 'zero':
        y[..., sl] = 0.0
    elif CTL['mode'] == 'batchmean':
        y[..., sl] = CTL['mean_t'][:T].unsqueeze(0).to(seg.dtype)
    elif CTL['mode'] == 'globalmean':
        y[..., sl] = CTL['mean_t'][:T].mean(0, keepdim=True).unsqueeze(0).to(seg.dtype)
    elif CTL['mode'] == 'donor':
        y[..., sl] = torch.roll(seg, 1, dims=0)
    elif CTL['mode'] == 'posshuffle':
        perm = torch.randperm(T, device=y.device)
        y[..., sl] = seg[:, perm]
    return (y,) + tuple(args[1:])


@torch.no_grad()
def ce_split(blocks, is_rare):
    tot = 0.0; n = 0; tr = 0.0; nr = 0; tf = 0.0; nf = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
        rm = is_rare[tgt]
        tot += float(ce_tok.sum()); n += tgt.shape[0]
        tr += float(ce_tok[rm].sum()); nr += int(rm.sum())
        tf += float(ce_tok[~rm].sum()); nf += int((~rm).sum())
    return tot/n, tr/max(nr, 1), tf/max(nf, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks[:, 1:].to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    # pass 1: capture head output (c_proj input slice) -> per-position batch mean + norm curve
    segs = []
    def cap(mo, args): segs.append(args[0][..., HH*HD:(HH+1)*HD].detach().float())
    hc = H[L].attn.c_proj.register_forward_pre_hook(cap)
    for i in range(0, NSEQ, 8): fwd(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    hc.remove()
    S = torch.cat(segs, 0)                      # N, T, HD
    CTL['mean_t'] = S.mean(0)                   # T, HD
    norm_t = S.norm(dim=-1).mean(0)             # T
    ramp_ratio = float(norm_t[-32:].mean()/norm_t[8:40].mean())
    del S, segs

    hk = H[L].attn.c_proj.register_forward_pre_hook(head_hook)
    CTL['mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    res = {}
    for mode in ['zero', 'batchmean', 'globalmean', 'donor', 'posshuffle']:
        CTL['mode'] = mode
        c, cr, cf = ce_split(blocks, is_rare)
        res[mode] = {'cost': round(c-base, 4), 'rare_cost': round(cr-base_r, 4), 'freq_cost': round(cf-base_f, 4)}
        CTL['mode'] = None
        print(f"{mode:>11}: cost {res[mode]['cost']} | rare {res[mode]['rare_cost']} | freq {res[mode]['freq_cost']}", flush=True)
    hk.remove()

    out = {'base_ce': round(base, 4), 'conditions': res, 'ramp_final_over_early': round(ramp_ratio, 3),
           'norm_curve_sample': [round(float(v), 2) for v in norm_t[::16]]}
    out['pred_a_generic_positional'] = bool(res['batchmean']['cost'] < 0.05 and res['globalmean']['cost'] > 0.3
                                            and ramp_ratio > 1.5)
    out['pred_b_position_over_document'] = bool(res['posshuffle']['cost'] > res['batchmean']['cost'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ramp final/early {ramp_ratio:.3f} | norm curve {out['norm_curve_sample']}", flush=True)
    print(f"pred_a generic-positional {out['pred_a_generic_positional']} | pred_b position>document {out['pred_b_position_over_document']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
