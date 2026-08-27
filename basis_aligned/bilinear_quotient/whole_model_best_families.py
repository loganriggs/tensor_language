# whole_model_best_families: THE BEST WHOLE-MODEL PROGRAM THIS ARC CAN BUILD
#
# §1694 substituted all thirty-six sites and reached 50.94% of a 5.5684-nat joint stake, using
# the SIMPLEST family on each side: a plain linear map at every MLP, a lag-1 map at every
# attention write. Both halves have known better families that the joint run did not use:
#
#   MLP side  (§1672): token tables at mlp0-2 with linear maps at mlp3-17 beat pure linear.
#                      mlp0 and mlp1 want a table, mlp2 is indifferent, mlp3 actively does not.
#   ATTN side (§1686/§1687): the lag set (1,2,4,8) reaches 68.05% against lag-1's 56.26%, and
#                      the positional description saturates there -- lag 64 adds 2.03 points.
#
# So the question §1694 leaves is whether the joint program improves by the amount the halves
# suggest, or whether compounding eats the upgrade. That is the culminating pricing number for
# this arc: the best whole-model program it can build, and what it costs.
#
# Compilation is interleaved bottom-up exactly as §1694 -- within block L, attn_L against
# everything substituted below it, installed, then mlp_L with attn_L also substituted. §1668 is
# the reason: independently-fitted programs installed together returned -42.99%.
#
# CONTROLS, three of them, because §1695 showed a ceiling is a ratio of three numbers and
# pinning one is not pinning the others:
#   - the BASELINE ce_live must reproduce 3.29205 (the §1695 failure, now asserted);
#   - the SIMPLE arm (linear everywhere, lag-1 everywhere) must reproduce §1694's 50.94%;
#   - the joint stake must reproduce §1694's 5.5684 nats, since it is the same condition.
#
# Registered predictions:
#   pred_a THE UPGRADE SURVIVES COMPOUNDING: the best-families joint program beats §1694's
#          50.94% by >= 8 percentage points. The halves gained 11.8 (attention, 56.26 -> 68.05)
#          and ~0 to 3.5 (MLP, depending on which figure), so a joint gain far below 8 would
#          mean compounding absorbs upgrades rather than passing them through.
#   pred_b COMPOUNDING PERSISTS: the joint ceiling still falls below the attention half's own
#          68.05%. If the joint program beats a half-ceiling, the joint and half conditions are
#          not measuring what I think they are.
#   pred_c ALL THREE CONTROLS HOLD: baseline within 1e-3 of 3.29205, simple arm within 1 point
#          of 50.94%, joint stake within 0.01 nats of 5.5684.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_best_families_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1694_SIMPLE = 0.5094
S1694_JOINT_STAKE = 5.5684
S1687_ATTN_BEST = 0.6805
S1672_MLP_TABLE_SITES = (0, 1, 2)
ATTN_LAGS = (1, 2, 4, 8)
S1683_CE_LIVE = 3.29205
CFG = {'lags': (1,), 'tables': ()}
STATE = {}
SEENREF = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def lagged(x):
    """[x_t, x_{t-l} for l in CFG['lags']], zero-padded at the start (§1685/§1686)."""
    lags = CFG['lags']
    parts = [x] + [torch.cat([torch.zeros_like(x[:, :l]), x[:, :-l]], dim=1) for l in lags]
    return torch.cat(parts, dim=-1).reshape(-1, D * (1 + len(lags)))


def table_hook(tbl):
    def hook(mod, args, out):
        sub = tbl[STATE['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def mlp_const_hook(c):
    def hook(mod, args, out):
        return c.to(out.dtype).expand_as(out)
    return hook


def attn_const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mlp_prog_hook(W):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def attn_prog_hook(W):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = (lagged(args[0]) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def install(prog):
    hs = []
    for (kind, L), W in prog.items():
        if kind == 'mlp':
            hs.append(H[L].mlp.register_forward_hook(
                table_hook(W) if L in CFG['tables'] else mlp_prog_hook(W)))
        else:
            hs.append(H[L].attn.register_forward_hook(attn_prog_hook(W)))
    return hs


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def fit_table(rows, L, prog):
    """Per-token mean of mlp_L's output with the stack below substituted (§1661 hybrid)."""
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert float(c.sum()) > 0, f'mlp{L}: no token counts'
    sn = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    return tbl


@torch.no_grad()
def fit_site(rows, kind, L, prog):
    if kind == 'mlp' and L in CFG['tables']:
        return fit_table(rows, L, prog)
    din = D if kind == 'mlp' else D * (1 + len(CFG['lags']))
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = (args[0].reshape(-1, D) if kind == 'mlp' else lagged(args[0])).double()
        y = (out if kind == 'mlp' else (out[0] if isinstance(out, tuple) else out))
        A.add_(x.T @ x); B.add_(x.T @ y.reshape(-1, D).double()); n['v'] += x.shape[0]
        return None
    tgt = H[L].mlp if kind == 'mlp' else H[L].attn
    sweep(rows, hooks=install(prog) + [tgt.register_forward_hook(collect)])
    assert n['v'] > 0, f'{kind}{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows, kinds):
    """Interleaved bottom-up: within block L, attn_L then mlp_L (§1669)."""
    prog = {}
    for L in ALL18:
        for kind in ('attn', 'mlp'):
            if kind in kinds:
                prog[(kind, L)] = fit_site(rows, kind, L, prog)
    return prog


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, seen, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    ev = load(EVAL_ROWS)
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    cl = ce(ev, seen)
    assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
        f'baseline CE {cl:.5f} disagrees with the known live CE {S1683_CE_LIVE:.5f} (§1695)')

    hs = [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
          for L in ALL18]
    hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
           for L in ALL18]
    cc = ce(ev, seen, hooks=hs)
    st = cc - cl
    print(f'WHOLE MODEL, BEST FAMILIES | interleaved bottom-up over 36 sites | CE live '
          f'{cl:.5f} | joint stake {st:.4f} nats (§1694 {S1694_JOINT_STAKE:.4f})', flush=True)
    print(f'  §1694 simple families: {S1694_SIMPLE:.2%} | §1687 attention half-ceiling at lags '
          f'(1,2,4,8): {S1687_ATTN_BEST:.2%}', flush=True)

    arms = {}
    for name, lags, tables in (('simple', (1,), ()),
                               ('best', ATTN_LAGS, S1672_MLP_TABLE_SITES)):
        CFG['lags'], CFG['tables'] = lags, tables
        prog = compile_stack(fit, ('mlp', 'attn'))
        ct = ce(ev, seen, hooks=install(prog))
        arms[name] = {'lags': list(lags), 'table_sites': list(tables),
                      'sites': len(prog), 'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5)}
        print(f'  {name:7s} lags {str(list(lags)):12s} tables {str(list(tables)):10s} '
              f'({len(prog)} sites) CEILING {arms[name]["ceiling"]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    simple, best = arms['simple']['ceiling'], arms['best']['ceiling']
    assert abs(best - simple) > 1e-6, f'both arms identical -- the family switch is a no-op'

    pa = (best - S1694_SIMPLE) >= 0.08
    pb = best < S1687_ATTN_BEST
    pc = (abs(cl - S1683_CE_LIVE) <= 1e-3 and abs(simple - S1694_SIMPLE) <= 0.01
          and abs(st - S1694_JOINT_STAKE) <= 0.01)

    print(f'\n  BEST families {best:.2%} vs §1694 simple {S1694_SIMPLE:.2%} '
          f'({best - S1694_SIMPLE:+.2%}) -> upgrade survives compounding {pa}', flush=True)
    print(f'  still below the attention half-ceiling {S1687_ATTN_BEST:.2%} -> compounding '
          f'persists {pb}', flush=True)
    print(f'  CONTROLS baseline {cl:.5f} | simple arm {simple:.2%} vs {S1694_SIMPLE:.2%} | '
          f'stake {st:.4f} vs {S1694_JOINT_STAKE:.4f} -> {pc}', flush=True)
    print(f'  nats recovered: {best * st:.4f} of {st:.4f}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'best_families': {'attn_lags': list(ATTN_LAGS),
                                        'mlp_table_sites': list(S1672_MLP_TABLE_SITES),
                                        'source': '§1672 (mlp0-2 want tables) and §1686/§1687 '
                                                  '(lags 1,2,4,8 saturate the positional family)'},
                      'compilation': 'INTERLEAVED bottom-up, attn_L then mlp_L within each block (§1669)',
                      'controls': ['baseline ce_live must reproduce 3.29205 (§1695)',
                                   'simple arm must reproduce §1694 50.94%',
                                   'joint stake must reproduce §1694 5.5684 nats'],
                      'coverage': 'hybrid, mask pinned to n96_skip80', 'scoring': 'covered positions only',
                      'v1_scope': 'attention v1 passed through unchanged (§1682, §1684)',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1694_simple': S1694_SIMPLE, 's1687_attn_best': S1687_ATTN_BEST},
           'ce_live': round(cl, 5), 'joint_stake': round(st, 5), 'arms': arms,
           'gain_over_simple': round(best - S1694_SIMPLE, 5),
           'nats_recovered': round(best * st, 5),
           'predictions': {'pred_a_upgrade_survives_ge_8pts': bool(pa),
                           'pred_b_compounding_persists': bool(pb),
                           'pred_c_all_three_controls': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
