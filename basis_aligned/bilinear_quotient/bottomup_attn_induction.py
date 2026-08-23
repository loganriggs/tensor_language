"""BOTTOM-UP push for attn2-4 (§1043: local-window recovery only 0.49-0.64 -- they route BEYOND a 4-token window,
consistent with induction copying from far back, §877). Add (a) a WIDER window (5 prev) and (b) an INDUCTION-COPY
feature: the embedding of the token that FOLLOWED the current token's previous occurrence ("what came after this token
last time"). Fit a ridge map from [emb(cur), emb(prev1..5), emb(induction-target)] to each attn output; held-out
loss-recovery per module. Tests whether front attn2-4 are understood as local-window + induction routing (toward 90%).

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-feature null ~0/negative; induction-target zero where the token is unseen.
  (a) INDUCTION LIFTS attn2-4: adding the wider window + induction feature raises attn2-4 loss-recovery ABOVE §1043's
      window-only (0.49-0.64), toward 90% -> front attn2-4 are local-window + induction routers;
  (b) attn5 stays LOW (broad pooler, not window/induction). Report recovery per module vs §1043 window-only."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_attn_induction_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [0, 1, 2, 3, 4, 5]; RIDGE = 1e3; NPREV = 5
REF_1043 = {'0': 0.944, '1': 0.825, '2': 0.643, '3': 0.488, '4': 0.615, '5': 0.089}
SUB = {'L': None, 'on': False, 'M': {}, 'gmean': {}, 'feat': None}


def induction_target(idx):
    # for each (b,t): token that followed the current token's previous occurrence in [0..t-1]; else -1
    B, T = idx.shape; out = torch.full((B, T), -1, dtype=torch.long, device=DEV)
    ic = idx.cpu().numpy()
    o = out.cpu().numpy()
    for b in range(B):
        last = {}
        row = ic[b]
        for t in range(T):
            tok = int(row[t])
            if tok in last:
                s = last[tok]
                if s+1 < T: o[b, t] = int(row[s+1])
            last[tok] = t
    return torch.tensor(o, device=DEV)


def feat(idx):
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); B, T, _ = E.shape
    feats = [E]
    for k in range(1, NPREV+1):
        sh = torch.zeros_like(E); sh[:, k:] = E[:, :T-k]; feats.append(sh)
    it = induction_target(idx)
    itc = it.clamp_min(0)
    ind_emb = F.rms_norm(m.transformer.wte(itc), (D,)).float()
    ind_emb = ind_emb * (it >= 0).float().unsqueeze(-1)   # zero where unseen
    feats.append(ind_emb)
    return torch.cat(feats, -1)  # (B,T,(NPREV+2)*D)


FD = (NPREV+2)*D


def forward_logits(idx):
    SUB['feat'] = feat(idx)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    def h(mo, i_, o_):
        if not SUB['on'] or SUB['L'] != L: return None
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if SUB['M'][L] is None:
            ny = SUB['gmean'][L].view(1, 1, D).expand(B, T, D)
        else:
            f = SUB['feat'].reshape(-1, FD); f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
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
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); feats.append(feat(idx).reshape(-1, FD).cpu()); forward_logits(idx)
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
    A = F1.T @ F1 + RIDGE*torch.eye(FD+1, device=DEV)
    for L in LAYERS:
        Yl = Y[L].to(DEV); SUB['M'][L] = torch.linalg.solve(A, F1.T @ Yl); SUB['gmean'][L] = Yl.mean(0); del Yl
    hooks = [m.transformer.h[L].attn.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['on'] = False; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['on'] = True; SUB['L'] = L
        sM = SUB['M'][L]; SUB['M'][L] = None; ce_ma = ce(te); SUB['M'][L] = sM
        denom = max(ce_ma - ce_full, 1e-6); ce_f = ce(te); SUB['on'] = False
        rec = round(float((ce_ma - ce_f)/denom), 3)
        out['layers'][str(L)] = {'recovery_window_induction': rec, 'recovery_window_only_1043': REF_1043[str(L)],
                                 'gain': round(rec - REF_1043[str(L)], 3)}
        print(f"attn{L}: window+induction {rec} | §1043 window-only {REF_1043[str(L)]} | gain {out['layers'][str(L)]['gain']}", flush=True)
    for h in hooks: h.remove()
    mid = [out['layers'][str(L)]['recovery_window_induction'] for L in [2,3,4]]
    out['attn234_mean'] = round(float(np.mean(mid)), 3)
    out['pred_a_induction_lifts'] = bool(all(out['layers'][str(L)]['gain'] > 0.05 for L in [2,3,4]))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"attn2-4 mean (window+induction) {out['attn234_mean']} | pred_a induction-lifts {out['pred_a_induction_lifts']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
