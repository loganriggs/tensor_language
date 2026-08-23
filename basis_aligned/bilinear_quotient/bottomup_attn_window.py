"""BOTTOM-UP 90%-per-module, next tier: the FRONT ATTENTION modules. §1035 scored attn by per-position token stand-ins
(attn0 0.81, attn1-4 0.3-0.5) -- but attention MOVES information from other positions, so the right stand-in is a
LOCAL WINDOW (current token + recent previous tokens), matching front attention's prev-token/induction/local routing
(§877). Fit a ridge map from [emb(cur), emb(prev1), emb(prev2), emb(prev3)] to each attn output, held-out, and measure
loss-recovery per module. This tests whether front attention is understood as a local-window function -- and where the
pooler (attn5) breaks that (it is broad, §1039).

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate = recovery 0; the local-window ridge map is genuine (a shuffled-window null would be ~0).
  (a) FRONT ATTN IS A LOCAL-WINDOW FUNCTION: attn0-4 reach HIGH loss-recovery (target > 0.7, toward 90%) with the
      local-window stand-in -- much higher than the per-position token stand-in (§1035 0.3-0.5) -> front attention is
      understood as local-window routing;
  (b) POOLER EXCEPTION: attn5 (the content pooler) stays LOW (<~0.3) -- a local window cannot capture a broad pool
      (consistent with §1039); report loss-recovery per front-attn module."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_attn_window_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [0, 1, 2, 3, 4, 5]; RIDGE = 1e3; NPREV = 3
SUB = {'L': None, 'on': False, 'M': {}, 'gmean': {}, 'feat': None, 'shuffle': False}


def winfeat(idx):
    # [emb(cur), emb(prev1..NPREV)] rms-normed, (B,T,(NPREV+1)*D)
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); B, T, _ = E.shape
    feats = [E]
    for k in range(1, NPREV+1):
        sh = torch.zeros_like(E); sh[:, k:] = E[:, :T-k]; feats.append(sh)
    return torch.cat(feats, -1)  # (B,T,(NPREV+1)*D)


def forward_logits(idx):
    SUB['feat'] = winfeat(idx)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    def h(mo, i_, o_):
        if not SUB['on'] or SUB['L'] != L: return None
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if SUB['M'][L] is None:  # mean-ablate mode
            ny = SUB['gmean'][L].view(1, 1, D).expand(B, T, D)
        else:
            f = SUB['feat'].reshape(-1, (NPREV+1)*D)
            if SUB['shuffle']: f = f[torch.randperm(f.shape[0], device=DEV)]
            f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            ny = (f1 @ SUB['M'][L]).reshape(B, T, D)
        return (ny.to(y.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(y.dtype)
    return h


@torch.no_grad()
def capture(blocks, layers):
    caps = {L: [] for L in layers}; feats = []; hs = []
    for L in layers:
        a = m.transformer.h[L].attn
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).cpu())
            return h
        hs.append(a.register_forward_hook(mk(L)))
    SUB['on'] = False
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); feats.append(winfeat(idx).reshape(-1, (NPREV+1)*D).cpu()); forward_logits(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in layers}, torch.cat(feats, 0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Y, F_tr = capture(tr, LAYERS); F_tr = F_tr.to(DEV)
    F1 = torch.cat([F_tr, torch.ones(F_tr.shape[0], 1, device=DEV)], 1)
    A = F1.T @ F1 + RIDGE*torch.eye((NPREV+1)*D+1, device=DEV)
    for L in LAYERS:
        Yl = Y[L].to(DEV); SUB['M'][L] = torch.linalg.solve(A, F1.T @ Yl); SUB['gmean'][L] = Yl.mean(0); del Yl
    hooks = [m.transformer.h[L].attn.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['on'] = False; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['on'] = True; SUB['L'] = L
        saveM = SUB['M'][L]; SUB['M'][L] = None; ce_ma = ce(te); SUB['M'][L] = saveM   # mean-ablate
        denom = max(ce_ma - ce_full, 1e-6)
        SUB['shuffle'] = False; ce_win = ce(te)
        SUB['shuffle'] = True; ce_sh = ce(te); SUB['shuffle'] = False
        SUB['on'] = False
        rw = round(float((ce_ma - ce_win)/denom), 3); rs = round(float((ce_ma - ce_sh)/denom), 3)
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma-ce_full,3), 'recovery_window': rw, 'recovery_shuffled_null': rs}
        print(f"attn{L}: meanabl {ce_ma-ce_full:.3f} | window-recovery {rw} | shuffled-null {rs}", flush=True)
    for h in hooks: h.remove()
    fr = [out['layers'][str(L)]['recovery_window'] for L in [0,1,2,3,4]]
    out['front_attn_mean_recovery'] = round(float(np.mean(fr)), 3); out['attn5_recovery'] = out['layers']['5']['recovery_window']
    out['pred_a_front_local_window'] = bool(out['front_attn_mean_recovery'] > 0.7)
    out['pred_b_pooler_exception'] = bool(out['attn5_recovery'] < 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front attn (0-4) mean window-recovery {out['front_attn_mean_recovery']} | attn5 {out['attn5_recovery']}", flush=True)
    print(f"pred_a front local-window {out['pred_a_front_local_window']} | pred_b pooler exception {out['pred_b_pooler_exception']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
