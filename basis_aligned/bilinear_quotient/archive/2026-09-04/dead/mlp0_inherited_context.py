# mlp0_inherited_context: IS MLP0'S UN-TABLEABLE RESIDUE COMPUTED BY MLP0, OR INHERITED
# FROM ATTN0? — a bottom-up MLP0 question, and one Codex is not asking.
#
# §1324/§1326: mlp0's token-table ceiling is 86.3% of a .799-nat stake, leaving .110
# nats that no token-indexed function can produce. §780: mlp0 computes the token's
# CLASS, ~23-dimensional, only 44% linearly predictable from the embedding. The open
# question in `registry/_mlp0_dossier` is what that .110-nat residue IS. It is not
# previous-token: `context_residual_results.json` puts the prev-token share at .2068
# against its own .2115 null (though at block 2, not here).
#
# §843 supplies the hypothesis: "attn0 builds the COPY-SOURCE: it writes the PREVIOUS
# token's identity into the stream." mlp0's input is the embedding PLUS attn0's write.
# So mlp0 may not be a contextual module at all -- it may be a TOKEN FUNCTION OF A
# CONTEXTUALISED INPUT. If so its residue is INHERITED from attention, not computed,
# and the right program for mlp0 is a table over a richer index rather than a
# context-reading map.
#
# TEST: measure mlp0's token-table ceiling twice.
#   (a) normally
#   (b) with attn0's output replaced by its OPTIMAL CONSTANT, so mlp0's input is the
#       embedding plus a fixed vector and carries no per-position context at all.
# The table is refitted under each condition, because a table fitted in one regime and
# applied in another measures transfer rather than ceiling.
#
# If the ceiling RISES under (b), the residue was inherited. If it does not move, mlp0
# computes its own context-dependence and the residue is genuinely mlp0's.
#
# APPLYING §1655's LESSON: the mean-ablation baseline uses the OPTIMAL CONSTANTS from
# opt_ablation_consts_all.pt (all 198 components, on disk since 02:30) rather than a
# constant I compute myself, and unseen tokens fall back to the position-weighted mean
# rather than a zero vector. Both were the bugs that made my discarded §1655 run
# disagree with §1326.
#
# Registered predictions:
#   pred_a THE RESIDUE IS AT LEAST PARTLY INHERITED: mlp0's table ceiling under
#          attn0-ablation exceeds its normal ceiling by >= 5 percentage points.
#   pred_b MLP0 IS NEARLY A PURE TOKEN FUNCTION ONCE CONTEXT IS FROZEN: the ablated
#          ceiling reaches >= 95%.
#   pred_c MANIPULATION CHECK -- the intervention actually reaches mlp0: ablating attn0
#          changes mlp0's output by a relative L2 of >= 5%. If freezing attn0 barely
#          moves mlp0, the comparison is vacuous regardless of what the ceilings say.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_inherited_context_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1326_CEILING = 0.863


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def attn0_freeze_hook(const):
    def hook(mod, args, out):
        if isinstance(out, tuple):
            y = out[0]
            return (const.to(y.dtype).expand_as(y),) + tuple(out[1:])
        return const.to(out.dtype).expand_as(out)
    return hook


@torch.no_grad()
def forward_collect(rows, freeze_attn0, const_a0, collect=True):
    """Run the model; optionally freeze attn0's write. Returns per-token sums for the
    mlp0 table, the position-weighted mean, and mlp0's output norm."""
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)
    sq = {'n': 0.0, 'sum': 0.0}
    cap = {}
    hs = [H[0].mlp.register_forward_hook(lambda mo, a, o: cap.__setitem__('o', o.float()))]
    if freeze_attn0:
        hs.append(H[0].attn.register_forward_hook(attn0_freeze_hook(const_a0)))
    try:
        for i in range(0, rows.shape[0], 8):
            idx = rows[i:i + 8, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            o = cap['o'].reshape(-1, D)
            if collect:
                t = idx.reshape(-1)
                s.index_add_(0, t, o)
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
            sq['sum'] += float((o ** 2).sum()); sq['n'] += o.shape[0]
    finally:
        for h in hs:
            h.remove()
    seen = c > 0
    gmean = s.sum(0) / c.sum()
    tbl = gmean.unsqueeze(0).repeat(50257, 1)          # unseen -> weighted mean (§1655 fix)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, (sq['sum'] / max(sq['n'], 1)) ** 0.5


@torch.no_grad()
def ce(rows, mode, freeze_attn0, const_a0, const_m0, tbl=None):
    """mode: 'live' | 'const' (optimal-constant ablate mlp0) | 'table'."""
    hs = []
    state = {}
    if freeze_attn0:
        hs.append(H[0].attn.register_forward_hook(attn0_freeze_hook(const_a0)))
    if mode == 'const':
        hs.append(H[0].mlp.register_forward_hook(
            lambda mo, a, o: const_m0.to(o.dtype).expand_as(o)))
    elif mode == 'table':
        hs.append(H[0].mlp.register_forward_hook(
            lambda mo, a, o: tbl[state['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)))
    tot, n = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            state['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            tot += float(e.sum()); n += e.numel()
    finally:
        for h in hs:
            h.remove()
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    const_a0 = K['attn0'].to(DEV).float(); const_m0 = K['mlp0'].to(DEV).float()
    print(f'mlp0 INHERITED CONTEXT | optimal constants from opt_ablation_consts_all.pt '
          f'(attn0, mlp0) | fit skip1200, eval skip7000', flush=True)

    out = {}
    norms = {}
    for label, freeze in (('normal', False), ('attn0_frozen', True)):
        tbl, nrm = forward_collect(fit, freeze, const_a0)
        norms[label] = nrm
        ce_live = ce(ev, 'live', freeze, const_a0, const_m0)
        ce_const = ce(ev, 'const', freeze, const_a0, const_m0)
        ce_tbl = ce(ev, 'table', freeze, const_a0, const_m0, tbl)
        stake = ce_const - ce_live
        ceil = (ce_const - ce_tbl) / stake if stake > 1e-6 else float('nan')
        out[label] = {'ce_live': round(ce_live, 5), 'ce_const_ablated': round(ce_const, 5),
                      'ce_table': round(ce_tbl, 5), 'stake': round(stake, 5),
                      'table_ceiling': round(ceil, 5), 'mlp0_output_rms': round(nrm, 4)}
        print(f'  {label:13s} stake {stake:7.4f} | CE live {ce_live:.5f} const {ce_const:.5f} '
              f'table {ce_tbl:.5f} | CEILING {ceil:7.2%}', flush=True)

    rel_change = abs(norms['attn0_frozen'] - norms['normal']) / max(norms['normal'], 1e-9)
    d_ceiling = out['attn0_frozen']['table_ceiling'] - out['normal']['table_ceiling']

    pa = d_ceiling >= 0.05
    pb = out['attn0_frozen']['table_ceiling'] >= 0.95
    pc = rel_change >= 0.05

    print(f'\n  mlp0 output RMS: normal {norms["normal"]:.4f} -> attn0-frozen '
          f'{norms["attn0_frozen"]:.4f}  (relative change {rel_change:.2%})', flush=True)
    print(f'  CEILING SHIFT: {out["normal"]["table_ceiling"]:.2%} -> '
          f'{out["attn0_frozen"]["table_ceiling"]:.2%}   delta {d_ceiling:+.2%}', flush=True)
    print(f'  (§1326 reports {S1326_CEILING:.1%} normally, on its own rows/protocol)', flush=True)
    print(f'  manipulation check -- attn0 freeze moves mlp0 output >= 5%: {pc}', flush=True)

    res = {'config': {'site': 0, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'ablation_constants': 'opt_ablation_consts_all.pt (§1655 fix)',
                      'unseen_token_fallback': 'position-weighted mean (§1655 fix)',
                      'table_refitted_per_condition': True,
                      's1326_normal_ceiling_other_rows': S1326_CEILING},
           'conditions': out, 'ceiling_delta': round(d_ceiling, 5),
           'mlp0_output_rms_relative_change': round(rel_change, 5),
           'predictions': {'pred_a_ceiling_rises_ge_5pts': bool(pa),
                           'pred_b_ablated_ceiling_ge_95': bool(pb),
                           'pred_c_manipulation_reaches_mlp0': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
