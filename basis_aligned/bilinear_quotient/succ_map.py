# succ_map: IS "+1" IN THE WEIGHTS? (§1276 mechanism.) Head 8.7 reads earlier digit
# tokens and the model then predicts the NEXT digit. Weights-only test: push each digit's
# raw rms(wte) code through 8.7's value pipeline (L8 c_v head-7 slice, value-residual mixed
# with the block-0 c_v code of the same token — both weights-computable) and its c_proj
# slice, then read the result against the unembedding: does digit d's delivered write push
# digit d+1's logits above d's own and d-1's?
#
# Caveat registered up front (sufficiency-style, as §1238): the real c_v input at L8 is the
# processed stream, not raw codes — success proves weights-readability; failure would not
# rule the mechanism out.
#
# Registered predictions:
#   pred_a SUCCESSOR SHIFT: for >= 5 of 8 digits d (1..8), mean unembedding logit of the
#          d+1 token set under 8.7's map exceeds BOTH d's own set and d-1's set.
#   pred_b CONTROL HEAD FLAT: head 8.1's map achieves this for <= 2 of 8.
#   pred_c RANK: d+1's set ranks top-3 among the nine digit sets for >= 4 of 8 digits
#          under 8.7 (and its mean rank is better than under 8.1).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'succ_map_results.json'
H = m.transformer.h


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    dig = {d: [] for d in range(1, 10)}
    for tok in range(50257):
        try:
            s = enc.decode([tok]).strip()
        except Exception:
            continue
        for d in range(1, 10):
            if s in (str(d), f"{d}.", f"{d})"):
                dig[d].append(tok)
    dig = {d: torch.tensor(v, device=DEV) for d, v in dig.items()}

    at8 = H[8].attn
    at0 = H[0].attn
    lam = float(at8.lamb)
    W_u = m.lm_head.weight.float()                             # (V, D)

    def head_map(h, toks):
        x = F.rms_norm(m.transformer.wte(toks), (D,))          # raw codes
        v8 = at8.c_v(x).view(-1, 9, 128)[:, h]                 # L8 value slice (raw-code approx)
        v0 = at0.c_v(x).view(-1, 9, 128)[:, h]                 # block-0 value slice (v1 approx)
        vv = (1 - lam) * v8 + lam * v0
        y = torch.zeros(vv.shape[0], 9, 128, device=DEV, dtype=vv.dtype)
        y[:, h] = vv
        w = at8.c_proj(y.reshape(-1, D))                       # delivered write (per token)
        return (w.float() @ W_u.T)                             # (n, V) logit image

    def eval_head(h):
        wins = 0; ranks = []
        per_d = {}
        for d in range(1, 9):
            lg = head_map(h, dig[d]).mean(0)                   # mean logit image of digit-d codes
            set_means = {dd: float(lg[dig[dd]].mean()) for dd in range(1, 10)}
            succ = set_means[d + 1]; own = set_means[d]; prev = set_means.get(d - 1, -1e9)
            ok = succ > own and succ > prev
            wins += int(ok)
            order = sorted(set_means.values(), reverse=True)
            rank = order.index(succ) + 1
            ranks.append(rank)
            per_d[d] = {'succ': round(succ, 4), 'own': round(own, 4),
                        'prev': round(prev, 4) if d > 1 else None, 'rank_succ': rank, 'ok': ok}
        return wins, ranks, per_d

    w87, r87, p87 = eval_head(7)
    w81, r81, p81 = eval_head(1)
    top3 = sum(1 for r in r87 if r <= 3)
    out = {'per_d_87': p87, 'per_d_81_summary': {'wins': w81, 'mean_rank': round(sum(r81) / 8, 2)},
           'wins_87': w87, 'mean_rank_87': round(sum(r87) / 8, 2), 'top3_count_87': top3,
           'pred_a_successor_shift': bool(w87 >= 5),
           'pred_b_control_flat': bool(w81 <= 2),
           'pred_c_rank': bool(top3 >= 4 and sum(r87) < sum(r81)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"8.7 wins {w87}/8 mean-rank {out['mean_rank_87']} top3 {top3} | 8.1 wins {w81}/8 mean-rank {out['per_d_81_summary']['mean_rank']}")
    for d, v in p87.items():
        print(f"d={d}: succ {v['succ']} own {v['own']} prev {v['prev']} rank {v['rank_succ']} ok {v['ok']}")
    print(f"pred_a {out['pred_a_successor_shift']} | pred_b {out['pred_b_control_flat']} | pred_c {out['pred_c_rank']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
