"""BOTTOM-UP transition (§1048): push the transition MLPs (mlp2/3/4, 0.65-0.72 with token+window §1045) by ADDING a bag-of-words (pooled-content) feature -- they sit where pooled content starts arriving (L3+ gatherers). Stand-in = token table + [local window + causal bag-of-words mean]. Predicts the bag lifts mlp2/3 toward 0.9 (they read early pooled content). [orig note] push the front MLPs below 90% (mlp0 0.67, mlp2/3/4) with a matched stand-in. The front MLP output is a
per-token table PLUS a local-context term (the front input is the token embedding + local attention, which §1043
showed is local-window). Stand-in = per-token TABLE(current) + ridge map of the LOCAL WINDOW [emb(cur),emb(prev1..3)]
fit on the table residual. Held-out loss-recovery per front MLP. Tests whether front MLPs reach ~90% as
token + local-window functions.

REGISTERED PREDICTIONS:
  (0) SANITY: token-table alone reproduces §1035-ish (mlp1 ~0.9, mlp0 ~0.67); shuffled window ~0 extra.
  (a) LOCAL WINDOW LIFTS FRONT MLPs: adding the local-window term raises mlp0 (and mlp2/3/4) ABOVE token-only, toward
      0.9 -> front MLPs are token + local-context functions;
  (b) report token-only vs token+window loss-recovery per front MLP."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_mlp_transition_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [2, 3, 4]; RIDGE = 1e3; NPREV = 3
SUB = {'L': None, 'mode': None, 'table': {}, 'M': {}, 'gmean': {}, 'feat': None, 'tokids': None}


def winfeat(idx):
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); B, T, _ = E.shape
    feats = [E]
    for k in range(1, NPREV+1):
        sh = torch.zeros_like(E); sh[:, k:] = E[:, :T-k]; feats.append(sh)
    num = torch.cumsum(E, 1); den = torch.arange(1, T+1, device=DEV, dtype=E.dtype).view(1, T, 1)
    feats.append(num/den)   # causal bag-of-words running mean of embeddings (pooled content)
    return torch.cat(feats, -1)


FD = (NPREV+2)*D


def forward_logits(idx):
    SUB['feat'] = winfeat(idx); SUB['tokids'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    def h(mo, i_, o_):
        if SUB['mode'] is None or SUB['L'] != L: return None
        o = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = o.shape
        if SUB['mode'] == 'mean':
            ny = SUB['gmean'][L].view(1, 1, D).expand(B, T, D)
        else:
            ny = SUB['table'][L][SUB['tokids'].reshape(-1)].reshape(B, T, D)
            if SUB['mode'] == 'table_window':
                f = SUB['feat'].reshape(-1, FD); f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
                ny = ny + (f1 @ SUB['M'][L]).reshape(B, T, D)
        return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
    return h


@torch.no_grad()
def capture(blocks, layers):
    caps = {L: [] for L in layers}; feats = []; toks = []; hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).cpu())
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    SUB['mode'] = None
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        feats.append(winfeat(idx).reshape(-1, FD).cpu()); toks.append(idx.reshape(-1).cpu()); forward_logits(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in layers}, torch.cat(feats, 0), torch.cat(toks, 0)


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
    V = int(m.lm_head.weight.shape[0]); nb = rows.shape[0]; ntr = int(0.6*nb)
    tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Y, Ftr, tok = capture(tr, LAYERS); Ftr = Ftr.to(DEV); tok = tok.to(DEV)
    for L in LAYERS:
        Yl = Y[L].to(DEV); gmean = Yl.mean(0)
        table = gmean.view(1, D).repeat(V, 1); sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        sums.index_add_(0, tok, Yl); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        seen = cnts >= 5; table[seen] = sums[seen]/cnts[seen].unsqueeze(1)
        resid = Yl - table[tok]
        F1 = torch.cat([Ftr, torch.ones(Ftr.shape[0], 1, device=DEV)], 1)
        M = torch.linalg.solve(F1.T @ F1 + RIDGE*torch.eye(FD+1, device=DEV), F1.T @ resid)
        SUB['table'][L] = table.half(); SUB['M'][L] = M; SUB['gmean'][L] = gmean; del Yl, resid
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['mode'] = None; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['L'] = L; SUB['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        SUB['mode'] = 'table'; ce_t = ce(te)
        SUB['mode'] = 'table_window'; ce_tw = ce(te)
        SUB['mode'] = None
        rt = round(float((ce_ma - ce_t)/denom), 3); rtw = round(float((ce_ma - ce_tw)/denom), 3)
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma-ce_full,3), 'recovery_token': rt, 'recovery_token_winbag': rtw, 'gain': round(rtw-rt,3)}
        print(f"mlp{L}: token {rt} | token+win+bag {rtw} | gain {out['layers'][str(L)]['gain']} (meanabl {ce_ma-ce_full:.3f})", flush=True)
    for h in hooks: h.remove()
    out['pred_a_bag_lifts'] = bool(out['layers']['2']['gain'] > 0.05 or out['layers']['3']['gain'] > 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a bag lifts transition MLPs: {out['pred_a_bag_lifts']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
