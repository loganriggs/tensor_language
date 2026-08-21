"""ATTN L5 OTHER VARIABLE (784: attn L5 reads position + a ~45% variable that is
NEITHER class nor position -- is that further variable a CLEAN LOW-RANK nameable
quantity, or the diffuse distributed remainder?). Take attn L5's input, project OFF
the token-class + position subspaces, and rank-sweep the REMAINING input subspace:
keep only the top-r residual directions and measure how much of attn L5's benefit
they recover. If a small r recovers the 45%, the "other variable" is low-rank
(nameable); if it needs many dims, it is diffuse. Also test data-stability of that
residual causal subspace (canonical vs fit-specific).

Runs on 256 rows (more data).

REGISTERED PREDICTIONS:
  (0) SANITY: removing class+position from L5's input leaves benefit (the 45%);
  (a) LOW-RANK NAMEABLE: a small residual rank (r <= 32) recovers most of the
      remaining benefit (>= 0.6 of the non-class-non-position part) AND that subspace
      is data-stable across two halves (overlap >= 0.6) -- so L5's further variable is
      a clean low-rank canonical quantity worth naming;
  (b) report residual-rank CE-recovery sweep + data-stability;
  ALT/NULL: if it needs many dims and is not data-stable, L5's other variable is the
      diffuse distributed remainder (no low-rank carrier), consistent with FINDINGS 1."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; L = 5
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_l5_other_results.json'
NEVAL = 256; MINCOUNT = 8; RTOK = 64; RPOS = 32; RS = [2, 8, 32, 128]
IN = {'U': None, 'op': None}; OUTABL = {'on': False}


def pre_hook(mo, args):
    if IN['op'] is None: return None
    x = args[0]; sh = x.shape; v = x.reshape(-1, D).float(); U = IN['U']
    v2 = (v @ U) @ U.T if IN['op'] == 'keep' else v - (v @ U) @ U.T
    return (v2.reshape(sh).to(x.dtype),) + tuple(args[1:])


def out_hook(mo, i_, o_):
    if not OUTABL['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture_in(rows, n):
    cap = []; toks = []; pos = []
    def pre(mo, args): cap.append(args[0].detach().float().reshape(-1, D))
    h = m.transformer.h[L].attn.register_forward_pre_hook(pre)
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def residual_dirs(Ain, Ucp):
    R = Ain - (Ain @ Ucp) @ Ucp.T                        # input off class+position
    Rc = R - R.mean(0, keepdim=True)
    return torch.linalg.svd(Rc, full_matrices=False)[2]  # (k, D) residual principal dirs


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    hp = m.transformer.h[L].attn.register_forward_pre_hook(pre_hook)
    ho = m.transformer.h[L].attn.register_forward_hook(out_hook)
    Ain, toks, pos = capture_in(rows, NEVAL)
    Utok = mean_subspace(Ain, toks, RTOK); Upos = mean_subspace(Ain, pos, RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()

    IN['op'] = None; OUTABL['on'] = False; ce_full = ce_on(rows, NEVAL)
    OUTABL['on'] = True; ce_abl = ce_on(rows, NEVAL); OUTABL['on'] = False; ben = ce_abl - ce_full
    # benefit remaining after removing class+position from input (the "other" part)
    IN['op'] = 'remove'; IN['U'] = Ucp; ce_cp_removed = ce_on(rows, NEVAL); IN['op'] = None; IN['U'] = None
    other_frac = float((ce_cp_removed - ce_full)/max(ben, 1e-6))   # how much L5 still does with NO class/pos input... (keep complement)
    # residual principal dirs of the input off class+position
    Vres = residual_dirs(Ain, Ucp)
    def keeprec(U): IN['op'] = 'keep'; IN['U'] = U; c = ce_on(rows, NEVAL); IN['op'] = None; IN['U'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    sweep = {}
    for r in RS:
        Ur = Vres[:r].T.contiguous(); sweep[str(r)] = round(keeprec(Ur), 4)
        print(f'keep-only residual-rank {r}: CE-recovery {sweep[str(r)]}', flush=True)

    # data-stability of the residual subspace
    half = Ain.shape[0]//2
    Va = residual_dirs(Ain[:half], Ucp); Vb = residual_dirs(Ain[half:], Ucp)
    r = 32; dov = float(torch.linalg.svdvals(Va[:r] @ Vb[:r].T).mean())
    print(f'benefit {ben:.3f} | keep class+pos -> other-part {other_frac:.3f} | residual subspace data-stability (top-32) {dov:.3f}', flush=True)
    hp.remove(); ho.remove()

    p0 = ben > 0
    pa = sweep['32'] >= 0.6 and dov >= 0.6
    out = {'benefit': round(ben, 4), 'keep_class_pos_complement': round(other_frac, 4),
           'residual_rank_sweep': sweep, 'residual_data_stability': round(dov, 4),
           'pred_0': bool(p0), 'pred_a_lowrank_nameable': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n(a) L5 other-variable is LOW-RANK nameable (rank-32 >=0.6 & data-stable): {pa}", flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
