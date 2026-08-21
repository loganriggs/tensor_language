"""QK LAYER DECOMP (user: QK circuit as a per-layer mini-win, bottom-up,
input-focused). Each softmax-free head has TWO content QK bilinear forms on
the residual: M1_h = W_q[h].T @ W_k[h] and M2_h = W_q2[h].T @ W_k2[h]
(1152x1152, rank <= HD). The attention pattern is (x_q.T M1 x_k)(x_q.T M2
x_k) -- a product of two bilinear scores (682-685). These forms are the
INPUT-focused QK circuit: they decide which residual directions at the
query pair with which at the key. At LAYER 0 the residual IS rms_norm(the
embedding), so we can read each form's top mode out through the embedding as
"which query-TOKENS look for which key-TOKENS."

For each head, SVD each content form, take the top mode (query dir u, key dir
v), and list the tokens with the highest |embedding . u| (query side) and
|embedding . v| (key side). Content forms only (rotary/positional part is
modulation on top -- flagged, handled in a follow-up).

REGISTERED PREDICTIONS:
  (0) SANITY: each form is low-rank (rank <= HD=128) and its top mode
      captures a real fraction of its energy (top-1 singular / sum > 1/HD);
  (a) INTERPRETABLE: for several heads at layer 0, the top query-tokens and
      key-tokens form a human-readable pattern (e.g. punctuation-looks-for-X,
      or a token-class pairing) -- report the per-head token lists;
  (b) report, per head, both QK forms' top query/key tokens + effective rank;
  NULL: a RANDOM 1152x1152 form of the same rank has no interpretable token
      structure (its top-mode tokens are arbitrary) -- report for contrast."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD

D = 1152
LAYER = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_layer_decomp_results.json'
TOPK = 8


def eff_rank(sv):
    sv = sv[sv > 0]
    return float((sv.sum()**2)/(sv**2).sum())


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def head_form(Wq, Wk, h):
    q = Wq[h*HD:(h+1)*HD, :]        # (HD, 1152)
    k = Wk[h*HD:(h+1)*HD, :]
    return q.T @ k                  # (1152, 1152) content QK form


@torch.no_grad()
def top_tokens(vecdir, Enorm, k=TOPK):
    s = (Enorm @ vecdir).cpu().numpy()
    order = np.argsort(-s)
    pos = [d1(t) for t in order[:k]]; neg = [d1(t) for t in order[::-1][:k]]
    return pos, neg


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    blk = m.transformer.h[LAYER].attn
    Wq = blk.c_q.weight.data.float().to(DEV); Wk = blk.c_k.weight.data.float().to(DEV)
    Wq2 = blk.c_q2.weight.data.float().to(DEV); Wk2 = blk.c_k2.weight.data.float().to(DEV)
    E = m.transformer.wte.weight.data.float().to(DEV)
    Enorm = F.rms_norm(E, (D,))     # layer-0 input = rms_norm(embedding)

    heads = {}
    for h in range(NH):
        entry = {}
        for tag, Wqx, Wkx in [('QK1', Wq, Wk), ('QK2', Wq2, Wk2)]:
            M = head_form(Wqx, Wkx, h)
            U, S, Vh = torch.linalg.svd(M)
            er = eff_rank(S)
            qpos, qneg = top_tokens(U[:, 0], Enorm)      # query side, top mode
            kpos, kneg = top_tokens(Vh[0, :], Enorm)     # key side, top mode
            top_frac = float(S[0]/S.sum())
            entry[tag] = {'eff_rank': round(er,1), 'top_frac': round(top_frac,3),
                          'query_tokens+': qpos, 'query_tokens-': qneg,
                          'key_tokens+': kpos, 'key_tokens-': kneg}
        heads[h] = entry
        print(f'head {h}: QK1 er {entry["QK1"]["eff_rank"]:.0f}  q+ {entry["QK1"]["query_tokens+"][:4]}  '
              f'k+ {entry["QK1"]["key_tokens+"][:4]}', flush=True)

    # null: random same-rank form
    g = torch.Generator(device=DEV).manual_seed(0)
    Rq = torch.randn(HD, D, generator=g, device=DEV); Rk = torch.randn(HD, D, generator=g, device=DEV)
    Mr = Rq.T @ Rk; Ur, Sr, Vhr = torch.linalg.svd(Mr)
    null_q, _ = top_tokens(Ur[:, 0], Enorm)
    print(f'\nNULL random form top query tokens: {null_q[:5]}', flush=True)

    er0 = np.mean([heads[h]['QK1']['eff_rank'] for h in range(NH)])
    p0 = er0 <= HD
    out = {'layer': LAYER, 'heads': heads, 'null_query_tokens': null_q,
           'mean_QK1_eff_rank': round(float(er0),1), 'pred_0_lowrank': bool(p0),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nmean QK1 eff-rank {er0:.1f} (<= HD {HD}: {p0})', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
