"""Turn §1055's eyeballed content axes into REPRODUCIBLE QUANTITATIVE labels. §1055 read the top content PCA axes as
topic/register contrasts by inspecting extremal snippets; this quantifies them: correlate each top content direction's
per-position coordinate with measurable surface/register features of the trailing context (digit density, punctuation
density, uppercase ratio, mean token length, leading-space/word-rate). If the top content axes correlate strongly with
interpretable surface features, we have a reproducible characterization (not snippet-eyeballing); the variance NOT
captured by surface features is genuine semantic topic beyond register. Fast: decode each vocab id once to per-token
char stats, aggregate over a trailing window by cumulative sum.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-position feature correlations ~ 0.
  (a) AXES CARRY REGISTER: several top content PCs correlate with surface/register features at |r| clearly above the
      shuffled null (e.g. a numeric/formal axis, a punctuation/list axis) -> §1055's interpretation is quantitatively
      real, not eyeballing;
  (b) RESIDUAL SEMANTIC TOPIC: the best single-feature |r| per PC is well below 1 and much content variance is NOT
      explained by these surface features (multiple-regression R^2 of all features on the content coords is modest) ->
      the content is more than surface register (genuine topic). Report per-PC top feature correlations + regression R^2
      + shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
import tiktoken
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_axis_features_results.json'
NEVAL = 200; SEQ = 256; REF = [8, 10, 12]; K = 16; CTX = 12
import census_lib as cl
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def capture(idx):
    hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    fwd(idx)
    for h in hs: h.remove()


def vocab_feature_table(V, enc):
    # per-token char stats: [n_chars, n_digit, n_punct, n_upper, n_alpha, leading_space]
    F_ = np.zeros((V, 6), dtype=np.float32)
    for i in range(V):
        try: s = enc.decode([i])
        except Exception: s = ''
        if not s: F_[i] = [1, 0, 0, 0, 0, 0]; continue
        nc = len(s); nd = sum(c.isdigit() for c in s); nu = sum(c.isupper() for c in s)
        na = sum(c.isalpha() for c in s); npu = sum((not c.isalnum()) and (not c.isspace()) for c in s)
        F_[i] = [nc, nd, npu, nu, na, 1.0 if s[0] == ' ' else 0.0]
    return F_


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    enc = tiktoken.get_encoding('gpt2'); V = int(m.lm_head.weight.shape[0])
    blocks = rows[:, :SEQ].contiguous(); T = blocks.shape[1] - 1
    for L in REF: CAP[L] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.cpu()); capture(idx)
    ids = torch.cat(idsL, 0); flat = ids.reshape(-1).to(DEV)
    devsum = None
    for L in REF:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, flat, X); cnts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[flat]; devsum = dv if devsum is None else devsum + dv; del X; CAP[L] = []
    dev = devsum / len(REF); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    coords = (devc @ Vt[:K].T).cpu().numpy()                     # (N,K)

    # per-position trailing-window surface features via per-token table + cumsum
    vf = vocab_feature_table(V, enc)                            # (V,6)
    ids_np = ids.numpy()                                        # (Nseq,T)
    Nseq = ids_np.shape[0]
    perpos = vf[ids_np.reshape(-1)].reshape(Nseq, T, 6)         # (Nseq,T,6)
    csum = np.cumsum(perpos, axis=1)
    win = np.zeros_like(perpos)
    for t in range(T):
        lo = max(0, t - CTX + 1)
        win[:, t] = csum[:, t] - (csum[:, lo-1] if lo > 0 else 0)
    win = win.reshape(-1, 6)                                    # (N,6) window sums
    nchar = win[:, 0].clip(min=1)
    feats = {
        'digit_density': win[:, 1] / nchar,
        'punct_density': win[:, 2] / nchar,
        'upper_ratio': win[:, 3] / nchar,
        'alpha_ratio': win[:, 4] / nchar,
        'mean_tok_len': win[:, 0] / float(CTX),
        'word_rate': win[:, 5] / float(CTX),
    }
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean(); d = (a.std()*b.std())
        return float((a*b).mean()/d) if d > 0 else 0.0
    rng = np.random.default_rng(0); perm = rng.permutation(coords.shape[0])
    out = {'K': K, 'CTX': CTX, 'n_positions': int(coords.shape[0]), 'features': list(feats.keys()), 'per_pc': []}
    Fmat = np.stack([feats[f] for f in feats], 1)               # (N,F)
    Fc = Fmat - Fmat.mean(0); Fc = Fc / (Fc.std(0)+1e-8)
    for k in range(K):
        c = coords[:, k]
        rs = {f: round(corr(c, feats[f]), 3) for f in feats}
        rs_sh = max(abs(corr(c[perm], feats[f]) ) for f in feats)
        # multiple regression R^2 of all features on this PC
        beta, *_ = np.linalg.lstsq(Fc, (c-c.mean())/(c.std()+1e-8), rcond=None)
        pred = Fc @ beta; r2 = float(1 - ((((c-c.mean())/(c.std()+1e-8)) - pred)**2).mean())
        top = sorted(rs.items(), key=lambda kv: -abs(kv[1]))[:3]
        out['per_pc'].append({'pc': k, 'top_feature_corrs': top, 'surface_R2': round(r2, 3), 'shuffled_max_abs_r': round(rs_sh, 3)})
        print(f"PC{k}: top {top} | surfaceR2 {round(r2,3)} | shufmax {round(rs_sh,3)}", flush=True)
    out['mean_surface_R2'] = round(float(np.mean([p['surface_R2'] for p in out['per_pc']])), 3)
    out['pred_a_axes_carry_register'] = bool(any(max(abs(v) for _, v in p['top_feature_corrs']) > 0.2 for p in out['per_pc']))
    out['pred_b_residual_semantic'] = bool(out['mean_surface_R2'] < 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"mean surface-R2 {out['mean_surface_R2']} | pred_a register {out['pred_a_axes_carry_register']} | pred_b residual-semantic {out['pred_b_residual_semantic']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
