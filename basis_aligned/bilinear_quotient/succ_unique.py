# succ_unique: the §1278-logged uniqueness control — compute the digit-successor rank-1
# count (0..8) of EVERY head's weights map (its layer's c_v head-slice, value-residual mixed
# with block-0's same slice, through its c_proj slice, against the unembedding), all 162.
#
# Registered predictions:
#   pred_a 8.7 IS THE MAXIMUM across all 162 heads.
#   pred_b RUNNER-UP <= 5/8 (clear gap).
#   pred_c SUCCESSOR STRUCTURE IS RARE: median count across heads <= 1.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'succ_unique_results.json'
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
    W_u = m.lm_head.weight.float()
    at0 = H[0].attn
    codes = {d: F.rms_norm(m.transformer.wte(dig[d]), (D,)) for d in range(1, 10)}

    counts = {}
    for L in range(18):
        atL = H[L].attn
        lam = float(atL.lamb)
        for h in range(9):
            r1 = 0
            for d in range(1, 9):
                x = codes[d]
                vL = atL.c_v(x).view(-1, 9, 128)[:, h]
                v0 = at0.c_v(x).view(-1, 9, 128)[:, h]
                vv = (1 - lam) * vL + lam * v0
                y = torch.zeros(vv.shape[0], 9, 128, device=DEV, dtype=vv.dtype)
                y[:, h] = vv
                lg = (atL.c_proj(y.reshape(-1, D)).float() @ W_u.T).mean(0)
                set_means = [float(lg[dig[dd]].mean()) for dd in range(1, 10)]
                succ = set_means[d]                            # index d = digit d+1
                r1 += int(succ == max(set_means))
            counts[f'{L}.{h}'] = r1
        print(f"L{L}: {[counts[f'{L}.{h}'] for h in range(9)]}", flush=True)

    vals = sorted(counts.values(), reverse=True)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    med = vals[len(vals) // 2]
    out = {'counts_top10': ranked[:10], 'median': med,
           'c87': counts['8.7'],
           'pred_a_87_max': bool(top[0] == '8.7'),
           'pred_b_gap': bool(second[1] <= 5),
           'pred_c_rare': bool(med <= 1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top10 {ranked[:10]} | median {med} | 8.7 = {counts['8.7']}")
    print(f"pred_a max {out['pred_a_87_max']} | pred_b gap {out['pred_b_gap']} | pred_c rare {out['pred_c_rare']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
