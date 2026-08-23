"""CERTIFICATION SWEEP continues (§1130): the readout stand-in numbers (§1046: mlp17 0.85, mlp16 0.78 near-linear
reads) predate the held-out rule. Certify: fit each readout MLP's LINEAR stand-in (full-rank least-squares map
input→output + intercept) on half A, substitute on half B, CE recovery vs mean-ablation. Layers 15/16/17.
Also the token-augmented variant (linear map + held-out per-token output table on the residual) — §1090 said
mlp16's token part is big-but-redundant; does adding it help a CERTIFIED stand-in at all?

REGISTERED PREDICTIONS:
  (0) SANITY: full ~0; linear >= mean-ablation trivially.
  (a) CERTIFIED: held-out linear recoveries within 0.1 of §1046 (mlp17 >= 0.75, mlp16 >= 0.68, mlp15 reported)
      -> the near-linear-read numbers are real (linear maps generalize, unlike token tables);
  (b) LEAK: any drop > 0.2 -> §1046's numbers were fit-inflated; correct plainly;
  (c) TOKEN TERM ADDS NOTHING (the §1090 expectation): linear+token <= linear + 0.03 (the token part is
      redundant with the stream; a certified stand-in doesn't need it)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_standin_heldout_results.json'
NSEQ = 384; SEQ = 256; LAYERS = [15, 16, 17]
H = m.transformer.h
SUB = {'layer': -1, 'mode': None}
ST = {'W': {}, 'obar': {}, 'ttab': {}, 'seen': None, 'glob_out': {}}
CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        if SUB['mode'] == 'meanabl':
            return ST['obar'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        if SUB['mode'] == 'full': return None
        B, T, _ = x.shape
        xf = torch.cat([x.reshape(-1, D), torch.ones(B*T, 1, device=DEV)], 1)
        y = (xf @ ST['W'][L]).view(B, T, D)
        if SUB['mode'] == 'linear_token':
            tokk = CUR['tok'].reshape(-1)
            tadd = ST['ttab'][L][tokk].float()
            unseen = ~ST['seen'][tokk]
            if unseen.any(): tadd[unseen] = 0.0
            y = y + tadd.view(B, T, D)
        return y.to(o_.dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]
    V = int(m.lm_head.weight.shape[0])

    capI = {L: [] for L in LAYERS}; capO = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_):
                capI[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, A.shape[0], 8):
        idx = A[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    ST['seen'] = cn > 0
    for L in LAYERS:
        X = torch.cat(capI[L], 0); O = torch.cat(capO[L], 0); capI[L] = []; capO[L] = []
        Xf = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        ST['W'][L] = torch.linalg.lstsq(Xf, O).solution           # (D+1) x D
        resid = O - Xf @ ST['W'][L]
        tt = torch.zeros(V, D, device=DEV); tt.index_add_(0, tok, resid)
        ST['ttab'][L] = (tt / cn.clamp_min(1).unsqueeze(1)).half()
        ST['obar'][L] = O.mean(0)
        # in-fit R2 for reference
        r2 = 1 - float((resid**2).sum()/((O - O.mean(0))**2).sum())
        print(f"L{L}: linear fit R2 (train) {r2:.3f}", flush=True)
        del X, O, resid

    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['layer'] = -1; base = ce(B)
    res = {}
    for L in LAYERS:
        row = {}
        for mode in ['full', 'linear', 'linear_token', 'meanabl']:
            SUB['layer'] = L; SUB['mode'] = mode
            row[mode] = round(ce(B) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        abl = max(row['meanabl'], 1e-6)
        row_rec = {mo: round(1 - row[mo]/abl, 3) for mo in ['linear', 'linear_token']}
        res[str(L)] = {'costs': row, 'recov': row_rec}
        print(f"L{L}: linear recov {row_rec['linear']} | +token {row_rec['linear_token']} | mean-abl {row['meanabl']}", flush=True)
    for h in hs: h.remove()

    insample = {'16': 0.78, '17': 0.85}
    out = {'base_ce': round(base, 4), 'per_layer': res,
           'drop_vs_1046': {L2: round(insample[L2] - res[L2]['recov']['linear'], 3) for L2 in insample}}
    out['pred_a_certified'] = bool(res['17']['recov']['linear'] >= 0.75 and res['16']['recov']['linear'] >= 0.68
                                   and max(out['drop_vs_1046'].values()) <= 0.1)
    out['pred_b_leak'] = bool(max(out['drop_vs_1046'].values()) > 0.2)
    tok_gain = max(res[str(L)]['recov']['linear_token'] - res[str(L)]['recov']['linear'] for L in LAYERS)
    out['pred_c_token_redundant'] = bool(tok_gain <= 0.03)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"drops vs §1046: {out['drop_vs_1046']} | max token gain {tok_gain:+.3f}", flush=True)
    print(f"pred_a certified {out['pred_a_certified']} | pred_b leak {out['pred_b_leak']} | pred_c token-redundant {out['pred_c_token_redundant']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
