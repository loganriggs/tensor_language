# bilin12_signs: family side of §1238-40. Architectural fact stated first: bilin12's
# pattern = (q·k/HD)^2 row-normalized and swiglu18's = softmax — BOTH non-negative by
# construction. bilin18 (unnormalized PRODUCT of two branches) is the only sibling whose
# patterns can carry a sign: the §1239-40 anti-matcher design is architecturally exclusive
# to it. Empirical question here: do bilin12's matcher heads (L2 H1/H3, §1218) still show
# a weights-readable same-token criterion — necessarily through MAGNITUDE (s^2), with the
# pre-square s free to use either sign?
#
# Instrument: §1238's, adapted — 512 real-corpus tokens' rms(wte) codes through L2's real
# q/k pipelines (rotary at 200 vs 72); per head, AUC of same-token vs different-token on
# s^2 (the pattern's actual monotone quantity pre-normalization).
#
# Registered predictions:
#   pred_a WEIGHTS-READABLE HERE TOO: AUC(2.1) and AUC(2.3) >= 0.75 on s^2 (positive
#          direction — squared magnitude, no anti option).
#   pred_b STATION-SPECIFIC: mean AUC over L2's other four heads <= 0.65.
#   pred_c PRE-SQUARE SIGN LOGGED, one-sided claim only: for each matcher, same-token s has
#          a consistent sign (>= 80% agreement) — the branch is signed even though the
#          pattern cannot be.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin12_signs_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NTOK = 512; QPOS = 200; KPOS = 72


@torch.no_grad()
def layer_scores(L, toks):
    at = mdl.transformer.h[L].attn
    x = F.rms_norm(mdl.transformer.wte(toks), (D,))
    N = x.shape[0]
    dt = mdl.transformer.wte.weight.dtype
    cos_t, sin_t = rope_tables(QPOS + 1, HD, DEV, dt, 'bf16')
    cos_t, sin_t = cos_t[None, :, None, :], sin_t[None, :, None, :]

    def pipe(lin, pos):
        z = F.rms_norm(lin(x).view(N, NH, HD), (HD,)).view(1, N, NH, HD)
        return apply_rot(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])

    q = pipe(at.c_q, QPOS); k = pipe(at.c_k, KPOS)
    s = torch.einsum('bqhd,bkhd->hqk', q.float(), k.float()) / HD
    return s


def auc(S):
    N = S.shape[0]
    same = S.diag(); a = []
    for i in range(N):
        row = S[i]; diff = torch.cat([row[:i], row[i + 1:]])
        a.append(float((same[i] > diff).float().mean() + 0.5 * (same[i] == diff).float().mean()))
    return sum(a) / N


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    V12 = int(mdl.lm_head.weight.shape[0])
    rows = cl.fineweb_rows(8)[:, :256].reshape(-1).clamp_max(V12 - 1)
    uniq = torch.unique(rows)
    g = torch.Generator().manual_seed(4)
    sel = uniq[torch.randperm(len(uniq), generator=g)[:NTOK]].to(DEV)
    s = layer_scores(2, sel)                    # (6,N,N)
    res = {}; signs = {}
    for h in range(NH):
        res[f'2.{h}'] = round(auc(s[h] ** 2), 4)
        d = s[h].diag()
        signs[f'2.{h}'] = {'pos_share': round(float((d > 0).float().mean()), 4),
                           'mean_s': round(float(d.mean()), 4)}
    matchers = [res['2.1'], res['2.3']]
    others = [res[f'2.{h}'] for h in range(NH) if h not in (1, 3)]
    sign_ok = all(max(signs[k]['pos_share'], 1 - signs[k]['pos_share']) >= 0.8
                  for k in ('2.1', '2.3'))
    out = {'model': 'bilin12', 'n_tokens': NTOK, 'auc_s2': res, 'same_token_sign': signs,
           'pred_a_weights_readable': bool(min(matchers) >= 0.75),
           'pred_b_station_specific': bool(sum(others) / len(others) <= 0.65),
           'pred_c_branch_signed': bool(sign_ok),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"AUC(s^2) {res}")
    print(f"signs {json.dumps(signs)}")
    print(f"pred_a {out['pred_a_weights_readable']} | pred_b {out['pred_b_station_specific']} | pred_c {out['pred_c_branch_signed']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
