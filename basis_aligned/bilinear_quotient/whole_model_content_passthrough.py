"""FOLLOW-UP to §1070: is the CONTENT the isolated gap? §1070 showed replacing all 36 modules compounds to 12%. Here
we replace everything EXCEPT the deep-middle content MLPs (mlp5-14), which PASS THROUGH their true (undistorted)
computation, and greedily fit the rest on the corrupted stream. If recovery jumps well above §1070's 12%, the understood
scaffold (grammar + attention routing + readout) composes acceptably once the load-bearing content flows through intact
-> the content is the isolated gap. If it stays low, compounding among the 'understood' modules is itself the binding
constraint.

REGISTERED PREDICTIONS:
  (a) CONTENT IS THE ISOLATED GAP: passing the deep-middle content MLPs through (replacing all else greedily) recovers
      MUCH more than §1070's all-36 number (0.124) -> the understood scaffold composes when content is preserved;
  (b) report recovery vs §1070, and the curve.
ORIGINAL §1070 docstring:
CAPSTONE: the honest END-TO-END whole-model understanding number. The bottom-up map (§1040-1069) scored each module
IN ISOLATION, but §1050 showed per-module stand-ins don't naively compose (fit on clean activations, they compound on
the corrupted stream). Fix with GREEDY sequential fitting (§1051): replace modules bottom-up, and fit each module's
matched stand-in ON THE CORRUPTED STREAM produced by the already-replaced modules. Matched stand-ins: attention = local
window [cur+3prev] of its (corrupted) residual input; MLP = per-current-token output table + linear of its input. Track
the end-to-end loss-recovery as we replace more of the model, to localize where recovery drops (expected: the content
band). Final point = the whole-model understanding number for these matched stand-ins, composition handled.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate-ALL = recovery 0; full = 1.
  (a) FRONT+READOUT WELL-CAPTURED, CONTENT IS THE GAP: the cumulative recovery curve stays high through the front, then
      DROPS across the deep-middle content MLPs (L5-14) -> the whole-model gap localizes to the content frontier;
  (b) HONEST WHOLE-MODEL NUMBER: report the final all-36-module greedy recovery + the curve + mean-ablate baseline.
      Expectation: moderate final recovery (simple window/table+linear stand-ins cannot capture the high-rank content),
      with the front/attention/readout portions recovering well."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_content_passthrough_results.json'
NEVAL = 200; SEQ = 256; NPREV = 3; RIDGE = 1e3
MODS = [(k, L) for L in range(18) for k in ('attn', 'mlp')]   # execution order within each block: attn then mlp
KEEP = {('mlp', L) for L in range(5, 15)}   # deep-middle content MLPs: pass through (never replaced)
REPL = [x for x in MODS if x not in KEEP]
G = {'active': set(), 'capt': None, 'tok': None, 'SM': {}, 'capI': None, 'capY': None}


def sub(kind, L): return getattr(m.transformer.h[L], kind)


def window(resin, W=NPREV):
    B, T, _ = resin.shape; feats = [resin]
    for k in range(1, W+1):
        sh = torch.zeros_like(resin); sh[:, k:] = resin[:, :T-k]; feats.append(sh)
    return torch.cat(feats, -1)


def apply_standin(kind, L, resin):
    sm = G['SM'][(kind, L)]; B, T, _ = resin.shape; tok = G['tok'].reshape(-1)
    if kind == 'attn':
        f = window(resin).reshape(-1, (NPREV+1)*D)
        f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
        return (f1 @ sm['M']).reshape(B, T, D)
    else:
        x = resin.reshape(-1, D); x1 = torch.cat([x, torch.ones(x.shape[0], 1, device=DEV)], 1)
        return (sm['table'][tok] + x1 @ sm['M']).reshape(B, T, D)


def hook(kind, L):
    def h(mo, i_, o_):
        resin = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if G['capt'] == (kind, L):
            G['capI'].append(resin.detach()); G['capY'].append(o.detach().float())
            return None
        if (kind, L) in G['active']:
            if G.get('mean_mode'):
                B, T, _ = o.shape; ny = G['SM'][(kind, L)]['gmean'].view(1, 1, D).expand(B, T, D)
            else:
                ny = apply_standin(kind, L, resin)
            return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
        return None
    return h


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); G['tok'] = idx
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def capture_module(kind, L, blocks):
    G['capt'] = (kind, L); G['capI'] = []; G['capY'] = []; toks = []
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); G['tok'] = idx; toks.append(idx.reshape(-1)); fwd(idx)
    G['capt'] = None
    return torch.cat(G['capI'], 0), torch.cat(G['capY'], 0), torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    hooks = [sub(k, L).register_forward_hook(hook(k, L)) for k, L in MODS]
    G['active'] = set()
    ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    # greedy fit each REPL module on the corrupted stream (content MLPs pass through), then activate it
    gmeans = {}
    for j, (kind, L) in enumerate(REPL):
        I, Y, tok = capture_module(kind, L, tr)
        Iw = I; Yf = Y.reshape(-1, D); gmeans[(kind, L)] = Yf.mean(0)
        if kind == 'attn':
            f = window(Iw).reshape(-1, (NPREV+1)*D); f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            A = f1.T @ f1 + RIDGE*torch.eye((NPREV+1)*D+1, device=DEV)
            G['SM'][(kind, L)] = {'M': torch.linalg.solve(A, f1.T @ Yf)}
        else:
            table = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
            table.index_add_(0, tok, Yf); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
            table = table / cnts.clamp_min(1).unsqueeze(1)
            resid = Yf - table[tok]; x = Iw.reshape(-1, D); x1 = torch.cat([x, torch.ones(x.shape[0], 1, device=DEV)], 1)
            A = x1.T @ x1 + RIDGE*torch.eye(D+1, device=DEV)
            G['SM'][(kind, L)] = {'M': torch.linalg.solve(A, x1.T @ resid), 'table': table}
        G['active'].add((kind, L))
        del I, Y, Iw, Yf
    allmods = list(REPL)
    for k, L in REPL: G['SM'][(k, L)]['gmean'] = gmeans[(k, L)]
    # end-to-end: all REPL stand-ins active, content MLPs (KEEP) pass through
    G['active'] = set(allmods); G['mean_mode'] = False; ce_all = ce(te)
    # baseline: mean-ablate the REPL modules (content still passes through)
    G['mean_mode'] = True; ce_ma = ce(te); G['mean_mode'] = False
    denom = max(ce_ma - ce_full, 1e-6)
    curve = []
    for npref in list(range(2, len(allmods)+1, 2)):
        G['active'] = set(allmods[:npref]); c = ce(te)
        curve.append({'modules_replaced': npref, 'last': f'{allmods[npref-1][0]}{allmods[npref-1][1]}',
                      'recovery': round(float((ce_ma - c)/denom), 3)})
        print(f"replaced {npref}/{len(allmods)} scaffold (up to {allmods[npref-1][0]}{allmods[npref-1][1]}): recovery {curve[-1]['recovery']}", flush=True)
    for h in hooks: h.remove()
    out = {'ce_full': round(ce_full, 4), 'ce_meanablate_scaffold': round(ce_ma, 4), 'ce_standin_scaffold': round(ce_all, 4),
           'ref_1070_all36_recovery': 0.124, 'ref_1070_ce_standin_all': 6.8314, 'ref_1070_ce_meanablate_all': 7.3615,
           'scaffold_recovery_content_passthrough': round(float((ce_ma - ce_all)/denom), 3),
           'recovery_vs_1070_denominator': round(float((7.3615 - ce_all)/(7.3615 - ce_full)), 3), 'curve': curve}
    out['pred_a_content_is_isolated_gap'] = bool(out['recovery_vs_1070_denominator'] > 0.124*2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"SCAFFOLD greedy recovery (content MLPs passed through) = {out['scaffold_recovery_content_passthrough']} | vs-1070-denominator {out['recovery_vs_1070_denominator']} (1070 all-36 was 0.124)", flush=True)
    print(f"ce_standin_scaffold {ce_all:.3f} vs 1070 all-36 {out['ref_1070_ce_standin_all']} | full {ce_full:.3f} | pred_a {out['pred_a_content_is_isolated_gap']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
