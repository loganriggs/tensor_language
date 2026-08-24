# window_repeat: the INDUCTION SHARE of the long-range budget (§1194 follow-on).
# Synthetic rows: tokens[128:256] = tokens[0:128] (verbatim repeat). At scored positions
# t >= 128 the repeat source is t-128 — outside EVERY window here (distance 128 needs W >= 129),
# so windowing destroys copy-reading entirely while natural text loses only 0.082 @W128 (§1180).
# Registered predictions:
#   pred_a REPEAT BASE IS EASY: full-context CE on the repeated half <= 1.5 nats (induction
#          working; natural rows ~3.25).
#   pred_b WINDOW KILLS COPYING: window cost on repeat rows @W64 >= 3x natural (>= 0.62).
#   pred_c NO ESCAPE AT 128: repeat-row cost @W128 >= 5x natural 0.082 (>= 0.41) — the
#          repeat's value is genuinely distance-128 reading, not local statistics.
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
OUT = PT + 'window_repeat_results.json'
NR = 24; WS = [32, 64, 128]; QSTART = 128; STRIDE = 2


@torch.no_grad()
def full_forward(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]                     # verbatim repeat; token 256 left as-is
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
           'natural_refs_1180': {'W32': 0.3787, 'W64': 0.2069, 'W128': 0.0816},
           'pred_a_repeat_easy': bool(CE['base'] <= 1.5),
           'pred_b_window_kills': bool(cost['W64'] >= 0.62),
           'pred_c_no_escape_128': bool(cost['W128'] >= 0.41),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"base CE {CE['base']} | costs {cost}")
    print(f"pred_a easy {out['pred_a_repeat_easy']} | pred_b kills {out['pred_b_window_kills']} | pred_c no-escape {out['pred_c_no_escape_128']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
