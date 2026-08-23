"""What do bilin18's two distinctive architectural features DO? (1) x0 RE-INJECTION: each block computes
x = lambdas[0]*x + lambdas[1]*x0 with lambdas ~= [6.09, 6.09] -- a ~50/50 remix of the running residual with the ORIGINAL
token embedding x0 at EVERY block. (2) VALUE-RESIDUAL: each attention uses v = (1-lamb)*v + lamb*v1 with lamb=0.5 --
half of BLOCK-0's value routed into every layer. Never directly characterized. Ablate each (x0: set lambdas[1]=0 all
blocks -> 100% running residual, no re-injection; value-residual: set lamb=0 all attn -> pure current value), measure CE
cost, per-layer importance, and whether each differentially affects CONTENT (rare/topical targets) vs GRAMMAR
(frequent/function targets, §1068 frequency split). Downstream rms_norm makes these direction/mix ablations, not scale
artifacts.

REGISTERED PREDICTIONS:
  (0) SANITY: restoring params recovers baseline CE exactly.
  (a) BOTH MATTER: removing x0 re-injection and removing the value-residual each raise CE substantially (>~0.2 nats);
  (b) DIFFERENTIAL ROLE: x0 re-injection (re-supplying token identity at depth) helps GRAMMAR/frequent-word prediction
      relatively more; the value-residual (routing early value content) helps CONTENT/rare-word prediction relatively
      more. Report global CE costs, per-layer x0 cost, and the rare-vs-frequent split for each."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'arch_features_results.json'
NEVAL = 200; SEQ = 256; NBIN = 4
H = m.transformer.h; NL = len(H)


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_tok_ce(blocks):
    ces = []; tgts = []
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        ces.append(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf]); tgts.append(tf)
    return torch.cat(ces, 0), torch.cat(tgts, 0)


@torch.no_grad()
def ce_mean(blocks):
    c, _ = per_tok_ce(blocks); return float(c.mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    # save originals
    orig_lam = [blk.lambdas.data.clone() for blk in H]
    orig_vl = [blk.attn.lamb.data.clone() for blk in H]
    def restore():
        for blk, l, vl in zip(H, orig_lam, orig_vl): blk.lambdas.data.copy_(l); blk.attn.lamb.data.copy_(vl)

    base_ce, base_tgt = per_tok_ce(blocks); base_mean = float(base_ce.mean())
    # frequency bins (rare->frequent) by target log-freq
    tgt_np = base_tgt.cpu().numpy(); freq = np.bincount(tgt_np, minlength=V).astype(np.float64); logf = np.log(freq[tgt_np]+1)
    order = np.argsort(logf); binsz = len(order)//NBIN
    bins = [order[b*binsz:(b+1)*binsz] if b < NBIN-1 else order[b*binsz:] for b in range(NBIN)]
    base_np = base_ce.cpu().numpy()

    def cond_split(name):
        c, _ = per_tok_ce(blocks); cnp = c.cpu().numpy()
        inc = [round(float((cnp[b]-base_np[b]).mean()), 4) for b in bins]
        return {'ce_mean': round(float(c.mean()), 4), 'cost': round(float(c.mean())-base_mean, 4),
                'rare_to_freq_loss_inc': inc, 'rare_over_freq': round(inc[0]/max(inc[-1], 1e-6), 2)}

    out = {'base_ce': round(base_mean, 4), 'lambdas0': round(float(orig_lam[0][0]), 3), 'lambdas1': round(float(orig_lam[0][1]), 3),
           'lamb': round(float(orig_vl[0]), 3), 'bins_rare_to_freq': NBIN, 'conditions': {}}
    # (b) no x0 re-injection
    for blk in H: blk.lambdas.data[1] = 0.0
    out['conditions']['no_x0_reinject'] = cond_split('no_x0'); restore()
    # (c) no value-residual
    for blk in H: blk.attn.lamb.data.fill_(0.0)
    out['conditions']['no_value_residual'] = cond_split('no_vresid'); restore()
    # (d) both
    for blk in H: blk.lambdas.data[1] = 0.0; blk.attn.lamb.data.fill_(0.0)
    out['conditions']['both_ablated'] = cond_split('both'); restore()
    # sanity: restore recovers baseline
    out['sanity_restore_ce'] = round(ce_mean(blocks), 4)
    # per-layer x0 ablation cost
    perlayer = {}
    for L in range(NL):
        H[L].lambdas.data[1] = 0.0; perlayer[str(L)] = round(ce_mean(blocks)-base_mean, 4); restore()
    out['per_layer_x0_cost'] = perlayer
    # per-layer value-residual ablation cost
    perlayer_v = {}
    for L in range(NL):
        H[L].attn.lamb.data.fill_(0.0); perlayer_v[str(L)] = round(ce_mean(blocks)-base_mean, 4); restore()
    out['per_layer_vresid_cost'] = perlayer_v
    out['pred_a_both_matter'] = bool(out['conditions']['no_x0_reinject']['cost'] > 0.2 and out['conditions']['no_value_residual']['cost'] > 0.2)
    out['pred_b_x0_grammar_vresid_content'] = bool(out['conditions']['no_x0_reinject']['rare_over_freq'] < out['conditions']['no_value_residual']['rare_over_freq'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"base {base_mean:.4f} | no-x0 cost {out['conditions']['no_x0_reinject']['cost']} | no-vresid cost {out['conditions']['no_value_residual']['cost']} | both {out['conditions']['both_ablated']['cost']}", flush=True)
    print(f"no-x0 rare/freq {out['conditions']['no_x0_reinject']['rare_over_freq']} | no-vresid rare/freq {out['conditions']['no_value_residual']['rare_over_freq']}", flush=True)
    print(f"per-layer x0 cost: {perlayer}", flush=True)
    print(f"per-layer vresid cost: {perlayer_v}", flush=True)
    print(f"sanity restore {out['sanity_restore_ce']} (== base {base_mean:.4f}) | pred_a {out['pred_a_both_matter']} pred_b {out['pred_b_x0_grammar_vresid_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
