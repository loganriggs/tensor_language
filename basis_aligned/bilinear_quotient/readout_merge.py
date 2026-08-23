"""THREAD A (readout merge): HOW are the grammar-head and content-tail streams combined into the final logits?
§1033/1034 established grammar owns the top-5 (head), content owns the tail (targets 25x deeper); §1046 showed the
readout MLPs are near-linear reads. Never traced: the MERGE itself. Hypothesis: the merge is ADDITIVE-LINEAR at the
final residual -- the pre-readout residual x = grammar carrier (per-token mean + class) + content deviation, and the
lm_head reads them as independent additive logit contributions (content shifts tail log-probs without moving the head).
Test at the final residual (input to the last rms_norm): content component c = U_c U_c^T (x - xbar[tok]) with U_c =
top-64 PCA of final-residual deviations. Interventions: (1) REMOVE c; (2) INTERCHANGE c across sequences (donor swap,
roll batch by 1); (3) remove a random-64 projection of the deviation (control). Metrics split by stream: HEAD = prob
mass on the base top-5 + argmax-change rate; TAIL = mean log-prob of rare targets (batch-frequency <= 2). Also test
LINEARITY directly: cosine between the actual logit change from removing c and the linear prediction -W_lm @ c_normed.

REGISTERED PREDICTIONS:
  (0) SANITY: base CE ~ canary (~3.4); random-64 removal costs much less than content-64 removal on the tail.
  (a) ADDITIVE MERGE / STREAM SEPARATION: removing c hurts TAIL (rare-target log-prob drop) >= 4x more than it
      shifts HEAD (top-5 mass change, relative terms); argmax-change rate under content removal < 10%
      -> the head is grammar-carried and survives content deletion; content adds tail mass additively;
  (b) INTERCHANGE TRANSPORTS TAIL ONLY: donor-swap moves rare-target log-probs toward the DONOR context (donor's
      rare targets gain, own rare targets lose) while top-5 mass changes stay small (same 4x separation);
  (c) NEAR-LINEAR READ: mean cosine between actual logit-delta from removing c and the linear map prediction
      (-lm_head @ c after norm correction) > 0.7 -> the merge is a linear read of the content component."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_merge_results.json'
NSEQ = 128; SEQ = 256; K = 64; RARE_MAX = 2


def fwd_resid(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x  # final residual, pre final-norm


def logits_from(x):
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    # pass 1: capture final residuals + build xbar + content basis
    X_all, tok_all = [], []
    for i in range(0, NSEQ, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous()
        X_all.append(fwd_resid(idx).float().reshape(-1, D)); tok_all.append(idx.reshape(-1))
    X = torch.cat(X_all, 0); tok = torch.cat(tok_all, 0); del X_all
    V = int(m.lm_head.weight.shape[0])
    xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
    dev = X - xbar[tok]; dev = dev - dev.mean(0)
    _, S, Vt = torch.linalg.svd(dev, full_matrices=False); U_c = Vt[:K].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    U_r = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    del dev, X

    # rare targets = tokens with batch frequency <= RARE_MAX
    tfreq = torch.zeros(V, device=DEV)
    tgt_all = blocks[:, 1:].to(DEV).reshape(-1)
    tfreq.index_add_(0, tgt_all, torch.ones_like(tgt_all, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    # pass 2: interventions per minibatch
    agg = {c: {'ce': 0.0, 'rare_lp': 0.0, 'n_rare': 0, 'top5_mass': 0.0, 'argmax_diff': 0.0, 'n': 0}
           for c in ['base', 'remove_content', 'interchange', 'remove_random']}
    lin_cos_sum = 0.0; lin_cos_n = 0
    W = m.lm_head.weight.float()  # V x D
    for i in range(0, NSEQ, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = fwd_resid(idx).float()
        d = x - xbar[idx]
        c = (d @ U_c) @ U_c.T
        r = (d @ U_r) @ U_r.T
        variants = {'base': x, 'remove_content': x - c, 'interchange': x - c + torch.roll(c, 1, dims=0),
                    'remove_random': x - r}
        lg_base = logits_from(variants['base'])
        p_base = F.softmax(lg_base, -1)
        top5 = p_base.topk(5, -1).indices; am_base = lg_base.argmax(-1)
        tf = tgt.reshape(-1); rare_mask = is_rare[tf]
        for name, xv in variants.items():
            lg = lg_base if name == 'base' else logits_from(xv)
            lp = F.log_softmax(lg.float(), -1)
            ce_tok = -lp.reshape(-1, V)[torch.arange(tf.shape[0], device=DEV), tf]
            a = agg[name]
            a['ce'] += float(ce_tok.sum()); a['n'] += tf.shape[0]
            a['rare_lp'] += float(-ce_tok[rare_mask].sum()); a['n_rare'] += int(rare_mask.sum())
            pm = F.softmax(lg, -1).gather(-1, top5).sum(-1)
            a['top5_mass'] += float(pm.sum())
            a['argmax_diff'] += float((lg.argmax(-1) != am_base).float().sum())
            if name == 'remove_content':  # linearity check vs -W @ c (norm-corrected per position)
                delta = (lg - lg_base).reshape(-1, V)
                scale = x.reshape(-1, D).norm(dim=-1, keepdim=True) / (D ** 0.5)
                pred = -(c.reshape(-1, D) / scale.clamp_min(1e-6)) @ W.T
                cn = F.cosine_similarity(delta, pred, dim=-1)
                lin_cos_sum += float(cn.sum()); lin_cos_n += cn.shape[0]
        del x, d, c, r, variants, lg_base, p_base

    res = {}
    npos = agg['base']['n']
    for name, a in agg.items():
        res[name] = {'ce': round(a['ce']/a['n'], 4), 'rare_lp': round(a['rare_lp']/max(a['n_rare'], 1), 4),
                     'top5_mass': round(a['top5_mass']/npos, 4), 'argmax_change': round(a['argmax_diff']/npos, 4)}
    b = res['base']
    tail_drop = b['rare_lp'] - res['remove_content']['rare_lp']          # nats lost on rare targets
    head_shift = abs(b['top5_mass'] - res['remove_content']['top5_mass']) / max(b['top5_mass'], 1e-6)
    tail_rel = tail_drop / max(abs(b['rare_lp']), 1e-6)
    out = {'K': K, 'n_rare_frac': round(agg['base']['n_rare']/npos, 4), 'conditions': res,
           'content_var_frac_top64': round(float((S[:K]**2).sum()/(S**2).sum()), 4),
           'tail_drop_nats': round(tail_drop, 4), 'head_top5_shift_rel': round(head_shift, 4),
           'tail_drop_rel': round(tail_rel, 4),
           'lin_cosine': round(lin_cos_sum/max(lin_cos_n, 1), 4),
           'interchange_rare_delta': round(res['interchange']['rare_lp'] - res['remove_content']['rare_lp'], 4),
           'random_tail_drop_nats': round(b['rare_lp'] - res['remove_random']['rare_lp'], 4)}
    out['pred_a_stream_separation'] = bool(tail_rel >= 4*head_shift and res['remove_content']['argmax_change'] < 0.10)
    out['pred_c_near_linear_read'] = bool(out['lin_cosine'] > 0.7)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for name in res: print(f"{name:>16}: CE {res[name]['ce']} | rare-lp {res[name]['rare_lp']} | top5 {res[name]['top5_mass']} | argmax-chg {res[name]['argmax_change']}", flush=True)
    print(f"tail drop {out['tail_drop_nats']} nats (rel {out['tail_drop_rel']}) vs head shift {out['head_top5_shift_rel']} | random-ctrl tail {out['random_tail_drop_nats']}", flush=True)
    print(f"lin cosine {out['lin_cosine']} | pred_a {out['pred_a_stream_separation']} | pred_c {out['pred_c_near_linear_read']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
