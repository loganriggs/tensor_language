"""REDTEAM of §1082 (registered there): the argmax-flip excess under content removal (38% vs random-64 control 14%)
is not yet content-SPECIFIC, because the content-64 removal deletes 76% of deviation variance while the random-64
control deletes ~6% — the flip excess could be mere perturbation size. Control here: NORM-MATCHED random removal --
remove a random-64 projection r scaled per-position to ||c|| (same perturbation magnitude, different direction).
Also a DEVIATION-matched control: remove the FULL deviation (x -> xbar[tok] + nothing), which deletes 100% of
deviation -- if argmax flips track variance removed, full-dev removal should flip MORE than content-64.

REGISTERED PREDICTIONS:
  (0) SANITY: content-removal numbers reproduce §1082 (~38% flips, tail drop ~1.7 nats).
  (a) CONTENT-SPECIFIC HEAD VOTE: norm-matched random removal flips the argmax MUCH less than content removal
      (< half) and costs far less tail log-prob -> §1082's flip rate is about the content DIRECTIONS, not the
      perturbation size;
  (b) if norm-matched random flips comparably, §1082's 'content votes among head candidates' was a magnitude
      artifact -- correct it plainly (the head is then perturbation-sensitive, not content-sensitive)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_merge_vmatch_results.json'
NSEQ = 128; SEQ = 256; K = 64; RARE_MAX = 2


def fwd_resid(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def logits_from(x):
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    X_all, tok_all = [], []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous()
        X_all.append(fwd_resid(idx).float().reshape(-1, D)); tok_all.append(idx.reshape(-1))
    X = torch.cat(X_all, 0); tok = torch.cat(tok_all, 0); del X_all
    V = int(m.lm_head.weight.shape[0])
    xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
    dev = X - xbar[tok]; dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); U_c = Vt[:K].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    U_r = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    del dev, X

    tfreq = torch.zeros(V, device=DEV)
    ta = blocks[:, 1:].to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    conds = ['base', 'remove_content', 'normmatch_random', 'remove_fulldev']
    agg = {c: {'ce': 0.0, 'rare_lp': 0.0, 'n_rare': 0, 'top5_mass': 0.0, 'argmax_diff': 0.0, 'n': 0} for c in conds}
    for i in range(0, NSEQ, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = fwd_resid(idx).float()
        d = x - xbar[idx]
        c = (d @ U_c) @ U_c.T
        r = (d @ U_r) @ U_r.T
        rmatch = r * (c.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        variants = {'base': x, 'remove_content': x - c, 'normmatch_random': x - rmatch,
                    'remove_fulldev': xbar[idx]}
        lg_base = logits_from(variants['base'])
        top5 = F.softmax(lg_base, -1).topk(5, -1).indices; am_base = lg_base.argmax(-1)
        tf = tgt.reshape(-1); rare_mask = is_rare[tf]
        for name, xv in variants.items():
            lg = lg_base if name == 'base' else logits_from(xv)
            lp = F.log_softmax(lg.float(), -1)
            ce_tok = -lp.reshape(-1, V)[torch.arange(tf.shape[0], device=DEV), tf]
            a = agg[name]
            a['ce'] += float(ce_tok.sum()); a['n'] += tf.shape[0]
            a['rare_lp'] += float(-ce_tok[rare_mask].sum()); a['n_rare'] += int(rare_mask.sum())
            a['top5_mass'] += float(F.softmax(lg, -1).gather(-1, top5).sum(-1).sum())
            a['argmax_diff'] += float((lg.argmax(-1) != am_base).float().sum())
        del x, d, c, r, variants, lg_base

    res = {}
    for name, a in agg.items():
        res[name] = {'ce': round(a['ce']/a['n'], 4), 'rare_lp': round(a['rare_lp']/max(a['n_rare'], 1), 4),
                     'top5_mass': round(a['top5_mass']/a['n'], 4), 'argmax_change': round(a['argmax_diff']/a['n'], 4)}
    fc = res['remove_content']['argmax_change']; fr = res['normmatch_random']['argmax_change']
    out = {'K': K, 'conditions': res,
           'flip_content': fc, 'flip_normmatch': fr, 'flip_fulldev': res['remove_fulldev']['argmax_change'],
           'tail_content': round(res['base']['rare_lp'] - res['remove_content']['rare_lp'], 4),
           'tail_normmatch': round(res['base']['rare_lp'] - res['normmatch_random']['rare_lp'], 4)}
    out['pred_a_content_specific'] = bool(fr < 0.5*fc and out['tail_normmatch'] < 0.5*out['tail_content'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for name in res: print(f"{name:>18}: CE {res[name]['ce']} | rare-lp {res[name]['rare_lp']} | top5 {res[name]['top5_mass']} | argmax-chg {res[name]['argmax_change']}", flush=True)
    print(f"flips: content {fc} | norm-matched random {fr} | full-dev {out['flip_fulldev']} | pred_a content-specific {out['pred_a_content_specific']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
