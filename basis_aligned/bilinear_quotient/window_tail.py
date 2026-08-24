# window_tail: §1180's curve tail — W in {160,192,224}, scored positions >= 224.
# Registered: (a) monotone continues; (b) W=192 <= 0.04; (c) decay per +64 tokens >= 1.8x.
# NOTE scored window differs from §1180 (positions >=224 so W=224 is a strict window):
# compare shapes, not absolute values, across the two runs.
#
# full_window_model: THE WHOLE MODEL AS A WINDOW FUNCTION (§1179 capstone).
#
# Selection is window-computable (+0.014, §1166); front MLPs 0-4 are window functions
# (0.004-0.014 each, §1177-79). The global question: run the ENTIRE model per-position on
# only its last W tokens — how much of total function is genuinely long-range at all?
# This is the values-side complement to the selection fold: it bounds everything —
# content pooling, induction, the sink (position 0 leaves the window once t > W).
#
# Method: for each scored position t >= 128, logits = full 18-block model on tokens
# [t-W+1 .. t] (exact model, truncated context); CE against the true next token; compared
# to the full-context base CE at the same positions. W ∈ {16, 32, 64, 128}. NOTE the sink:
# a window forward's position 0 is the window's first token, not the document's — the
# MLP4-at-pos-0 constant is manufactured at the WINDOW's start, which prior work says is
# content-generic (§1089: constant = cross-doc cos 0.998), so the sink should survive.
#
# Registered predictions:
#   pred_a monotone in W.
#   pred_b W=64 costs <= 0.15 nats at scored positions.
#   pred_c W=128 costs <= 0.05.
# Control: base = full-context CE at the same scored positions (same rows, same stride).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'window_tail_results.json'
NR = 24; WS = [160, 192, 224]; QSTART = 224; STRIDE = 1


@torch.no_grad()
def full_forward(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    qpos = list(range(QSTART, T, STRIDE))
    ce = {f'W{w}': 0.0 for w in WS}; ce['base'] = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lb = full_forward(idx).float()
        qp = torch.tensor(qpos, device=DEV)
        ce['base'] += float(F.cross_entropy(lb[:, qp].reshape(-1, lb.shape[-1]),
                                            tgt[:, qp].reshape(-1), reduction='sum'))
        for w in WS:
            # windows: for each scored position t, tokens [t-w+1 .. t]
            wins = torch.stack([idx[:, t - w + 1: t + 1] for t in qpos], 1)   # (B, Q, w)
            B, Q, _ = wins.shape
            flat = wins.reshape(B * Q, w)
            outs = []
            step = max(64, 4096 // w)
            for j in range(0, flat.shape[0], step):
                lw = full_forward(flat[j:j + step]).float()
                outs.append(lw[:, -1])
            lwin = torch.cat(outs, 0).reshape(B, Q, -1)
            ce[f'W{w}'] += float(F.cross_entropy(lwin.reshape(-1, lwin.shape[-1]),
                                                 tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qpos)
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {k: round(v / n, 4) for k, v in ce.items()}
    cost = {f'W{w}': round(CE[f'W{w}'] - CE['base'], 4) for w in WS}
    seq = [cost[f'W{w}'] for w in WS]
    out = {'n_rows': NR, 'scored_positions': f'{QSTART}..{T} stride {STRIDE}', 'ce': CE,
           'cost_vs_fullcontext': cost,
           'pred_a_monotone': bool(all(seq[j + 1] < seq[j] for j in range(len(seq) - 1))),
           'pred_b_w192': bool(cost['W192'] <= 0.04),
           'pred_c_decay': bool(cost['W160'] / max(cost['W224'], 1e-6) >= 1.8),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"base CE {CE['base']} | costs {cost}")
    print(f"pred_a monotone {out['pred_a_monotone']} | pred_b W192 {out['pred_b_w192']} | pred_c decay {out['pred_c_decay']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
