"""MDL PARSIMONY, CE-BASED (user: per-datapoint minimal description; and
"do CE loss for A-SVD always"). Per datapoint, rank components by their
first-order CE effect -- removing component k from datapoint i's mlp1 output
changes CE by ~ g_i . (c_ik u_k), so CE-relevance_ik = |c_ik| * |g_i . u_k|
where c_ik = o_i . u_k (coefficient), g_i = dCE/do_i (one backward pass),
u_k = component direction. Per-datapoint CODE LENGTH = # components (ranked
by CE-relevance) to reach 90% of that datapoint's total first-order CE
magnitude. This is CE-based, not L2/energy (737: energy ordering front-loads
loss-irrelevant massive dims).

Compare A-SVD vs weight-SVD vs random dictionary of mlp1's output (K=256).
Report mean per-datapoint code length, reuse (usage Gini), and VALIDATION #2:
correlation of code length with next-token ENTROPY (harder->longer) and with
target LOG-FREQ (frequent->shorter).

REGISTERED PREDICTIONS:
  (0) SANITY: code lengths in [1,K];
  (a) DIFFICULTY-TRACKING: per-datapoint CE-code length correlates POSITIVE
      with next-token entropy and NEGATIVE with target log-freq (|corr|>=0.15)
      for at least one basis -- the decomposition respects true datapoint
      complexity (simple=short, hard=long);
  (b) report mean code length + reuse + difficulty correlations per basis;
  NULL: random dictionary has flat/near-zero difficulty correlation."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mdl_parsimony_results.json'
NFIT = 48; NDATA = 48; K = 256; FRAC = 0.90


def asvd_fast(W, X, eps=1e-3):
    U, Sg, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * Sg; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps); B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps); B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID).cpu()))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def capture_out_grad_meta(rows, n):
    """mlp1 output O, CE-gradient G=dCE/dO, next-token entropy, target log-freq."""
    Os = []; Gs = []; ents = []; lfs = []
    V = m.lm_head.weight.shape[0]; freq = np.zeros(V)
    for r in range(n):
        for t in rows[r, 1:257].tolist(): freq[t] += 1
    logf = np.log(freq + 1.0)
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        store = {}
        def hk(mo, i_, o_):
            o2 = o_.detach().requires_grad_(True); store['o'] = o2; return o2
        h = m.transformer.h[LAYER].mlp.register_forward_hook(hk)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        h.remove()
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum')
        ce.backward()
        o = store['o']
        Os.append(o.detach().float().reshape(-1, D).cpu()); Gs.append(o.grad.float().reshape(-1, D).cpu())
        with torch.no_grad():
            p = lp.exp(); e = -(p*lp).sum(-1).reshape(-1).cpu().numpy(); ents.append(e)
            lfs.append(np.array([logf[t] for t in tgt.reshape(-1).cpu().numpy()]))
        m.zero_grad(set_to_none=True)
    return torch.cat(Os,0), torch.cat(Gs,0), np.concatenate(ents), np.concatenate(lfs)


def ce_code_lengths(O, G, U, frac=FRAC):
    C = (O @ U).numpy()            # coeff (N,K)
    GU = (G @ U).numpy()           # CE-grad alignment (N,K)
    R = np.abs(C * GU)             # per-datapoint CE-relevance (N,K)
    Rsort = -np.sort(-R, axis=1); cum = np.cumsum(Rsort, axis=1); tot = cum[:, -1:]+1e-12
    lengths = ((cum/tot) < frac).sum(1) + 1
    order = np.argsort(-R, axis=1); usage = np.zeros(U.shape[1])
    for i in range(R.shape[0]): usage[order[i, :lengths[i]]] += 1
    p = usage/(usage.sum()+1e-12); p = np.sort(p[p>0])
    gini = float(1 - 2*np.sum(p.cumsum())/ (p.sum()*len(p)) + 1/len(p)) if len(p) else 0.0
    return lengths, gini


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    allrows = cl.fineweb_rows(NFIT + NDATA)
    fit, dat = allrows[:NFIT], allrows[NFIT:NFIT+NDATA]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    Xg = capture_gate(fit, NFIT).to(DEV)
    A, _ = asvd_fast(W, Xg); Ua = (A[:, :K] / A[:, :K].norm(dim=0, keepdim=True)).cpu()
    Uw = torch.linalg.svd(W)[0][:, :K].cpu()
    g = torch.Generator().manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, K, generator=g))[0]

    O, Grad, ent, tlf = capture_out_grad_meta(dat, NDATA)
    res = {}
    for name, U in [('asvd', Ua), ('weight_svd', Uw), ('random', Ur)]:
        lengths, gini = ce_code_lengths(O, Grad, U)
        ce_ = float(np.corrcoef(lengths, ent)[0,1]); lf_ = float(np.corrcoef(lengths, tlf)[0,1])
        res[name] = {'mean_code_len': round(float(lengths.mean()),2), 'std': round(float(lengths.std()),2),
                     'usage_gini': round(gini,3), 'corr_len_entropy': round(ce_,3), 'corr_len_logfreq': round(lf_,3)}
        print(f'{name:11s}: mean-len {res[name]["mean_code_len"]:.1f}  gini {gini:.2f}  '
              f'corr(len,entropy) {ce_:+.3f}  corr(len,logfreq) {lf_:+.3f}', flush=True)
    def tracks(r): return r['corr_len_entropy'] >= 0.15 and r['corr_len_logfreq'] <= -0.15
    pa = tracks(res['asvd']) or tracks(res['weight_svd'])
    null_ok = abs(res['random']['corr_len_entropy']) < 0.15
    print(f'\n(a) a basis tracks difficulty: {pa}; NULL random flat: {null_ok}', flush=True)
    out = {'layer': LAYER, 'K': K, 'frac': FRAC, 'n_datapoints': len(ent), 'ce_based': True,
           'bases': res, 'pred_a_tracks_difficulty': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
