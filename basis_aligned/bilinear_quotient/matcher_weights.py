# matcher_weights: is the MATCHING CRITERION readable from the WEIGHTS + EMBEDDINGS alone?
# §1237: the matchers' identity substrate does not ride the v1 broadcast — it reaches
# L2/L3's q/k from the embedding/MLP path. Strongest weight-relative version: feed RAW
# per-token codes x = rms_norm(wte(t)) through a head's real q- and k-pipelines
# (c_q -> rms -> rotary at query position p, c_k -> rms -> rotary at p-128) and ask whether
# the bilinear score s1*s2 separates SAME-token pairs from DIFFERENT-token pairs.
#
# Sampling: 512 tokens drawn from the FineWeb frequency distribution (real corpus tokens);
# for each, score(same) = score(t at p vs t at p-128); score(diff) vs 511 others -> AUC.
# Query position p = 200 (rotary phases realistic). Heads scored: matchers 2.5, 3.8;
# auxiliary 3.1; fetchers 8.3, 8.4; sink 5.7; plus ALL heads of L2 and L3 (context: is the
# criterion station-specific in pure weights?).
#
# CAVEAT REGISTERED UP FRONT: the real q/k inputs at L2/L3 are processed stream states, not
# raw embeddings — a high AUC here is SUFFICIENT to show a weights-readable identity
# criterion; a low AUC would NOT rule it out (the criterion could live in MLP-transformed
# coordinates). Predictions are therefore one-sided.
#
# Registered predictions:
#   pred_a MATCHERS SEPARATE ON RAW CODES: AUC(2.5) and AUC(3.8) >= 0.80.
#   pred_b STATION-SPECIFIC: mean AUC over non-station heads of L2+L3 <= 0.65, and
#          AUC(5.7 sink) <= 0.6.
#   pred_c FETCHERS DON'T (their key is composed, not raw identity): AUC(8.3), AUC(8.4)
#          <= AUC(matcher min) - 0.1.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_weights_results.json'
NTOK = 512; QPOS = 200; KPOS = 72   # offset 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def head_scores(L, toks):
    """Bilinear score matrix S[i,j] = head's pattern value for query-token i at QPOS
    attending key-token j at KPOS, computed from raw rms(wte) codes. Returns (9,N,N)."""
    at = m.transformer.h[L].attn
    x = F.rms_norm(m.transformer.wte(toks), (D,))          # (N, D)
    N = x.shape[0]
    dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
    cos_t, sin_t = at.rotary(dummy)                        # (1, Tn, 1, 64)

    def pipe(lin, pos):
        z = F.rms_norm(lin(x).view(N, 9, 128), (128,)).view(1, N, 9, 128)
        return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])
    q = pipe(at.c_q, QPOS); k = pipe(at.c_k, KPOS)
    q2 = pipe(at.c_q2, QPOS); k2 = pipe(at.c_k2, KPOS)
    s1 = torch.einsum('bqhd,bkhd->hqk', q.float(), k.float()) / 128.0
    s2 = torch.einsum('bqhd,bkhd->hqk', q2.float(), k2.float()) / 128.0
    return s1 * s2


def auc_same_vs_diff(S):
    """S: (N,N) score matrix; same = diagonal, diff = off-diagonal. Rank-based AUC."""
    N = S.shape[0]
    same = S.diag()
    aucs = []
    for i in range(N):
        row = S[i]
        diff = torch.cat([row[:i], row[i + 1:]])
        aucs.append(float((same[i] > diff).float().mean() + 0.5 * (same[i] == diff).float().mean()))
    return sum(aucs) / N


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(8)[:, :256].reshape(-1)
    uniq = torch.unique(rows)
    g = torch.Generator().manual_seed(4)
    sel = uniq[torch.randperm(len(uniq), generator=g)[:NTOK]].to(DEV)

    res = {}
    for L in (2, 3, 5, 8):
        S = head_scores(L, sel)                            # (9,N,N)
        for h in range(9):
            res[f'{L}.{h}'] = round(auc_same_vs_diff(S[h]), 4)
        print(f"L{L}: " + " ".join(f"{h}:{res[f'{L}.{h}']:.3f}" for h in range(9)), flush=True)

    matchers = [res['2.5'], res['3.8']]
    non_station = [res[f'{L}.{h}'] for L in (2, 3) for h in range(9)
                   if (L, h) not in ((2, 5), (3, 8), (3, 1))]
    ns_mean = sum(non_station) / len(non_station)
    out = {'n_tokens': NTOK, 'qpos': QPOS, 'offset': QPOS - KPOS, 'auc': res,
           'nonstation_L23_mean': round(ns_mean, 4),
           'pred_a_matchers_separate': bool(min(matchers) >= 0.80),
           'pred_b_station_specific': bool(ns_mean <= 0.65 and res['5.7'] <= 0.6),
           'pred_c_fetchers_dont': bool(max(res['8.3'], res['8.4']) <= min(matchers) - 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"matchers 2.5={res['2.5']} 3.8={res['3.8']} | aux 3.1={res['3.1']} | fetchers 8.3={res['8.3']} 8.4={res['8.4']} | sink 5.7={res['5.7']} | ns mean {ns_mean:.3f}")
    print(f"pred_a {out['pred_a_matchers_separate']} | pred_b {out['pred_b_station_specific']} | pred_c {out['pred_c_fetchers_dont']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
