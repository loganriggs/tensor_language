"""LAYER 0, attention: what does attn0 WRITE, mechanistically? (finishes layer 0; robust decode).
attn0 (benefit 1.44) is claimed to build the "copy-source" (census name/induction circuit, item 10).
Test at the mechanism level what attn0's output encodes: PREVIOUS token identity (copy-source),
CURRENT token, or POSITION. Decode each from attn0's output via a linear probe (train/test split),
and compare to a shuffled-label null. Also probe attn0's INPUT (embedding) as a reference for how much
prev-token info attn0 ADDS (embedding has current token, not previous). This characterizes attn0's role
to hand to layer 1. (The full two-criteria (q·k1)(q2·k2) attention-pattern mechanism needs rotary-aware
reconstruction — flagged for next.)

REGISTERED PREDICTIONS:
  (0) SANITY: current-token decodes near-perfectly from the embedding (it IS the embedding);
  (a) COPY-SOURCE: PREVIOUS-token identity decodes from attn0 OUTPUT far better than from the embedding
      input and far above the shuffled null -> attn0 moves previous-token info into the stream (builds
      the copy-source), the mechanistic content of its 1.44-nat contribution;
  (b) report decode accuracy for prev/current/position from attn0 output vs input vs null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn0_function_results.json'
NEVAL = 240; TOPV = 200          # decode among the TOPV most frequent tokens (manageable classes)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows):
    attn = m.transformer.h[0].attn; Ins = []; Outs = []; toks = []; pos = []
    def pre(mo, args): Ins.append(args[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = attn.register_forward_pre_hook(pre); ho = attn.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape))
    hp.remove(); ho.remove()
    return torch.cat(Ins, 0), torch.cat(Outs, 0), np.concatenate([t.reshape(-1) for t in toks]), np.concatenate([p.reshape(-1) for p in pos])


def decode_acc(F_feat, labels, valid, ncls, seed=0):
    """multiclass linear probe (ridge via normal eqn on one-hot), train/test split, top-1 accuracy."""
    idx = np.where(valid)[0]
    rng = np.random.RandomState(seed); rng.shuffle(idx)
    ntr = int(0.7*len(idx)); tr, te = idx[:ntr], idx[ntr:]
    Xtr = F_feat[tr]; Xte = F_feat[te]
    ytr = labels[tr]; yte = labels[te]
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(ytr, device=DEV)] = 1.0
    A = Xtr.T @ Xtr + 1e2*torch.eye(D, device=DEV)
    Wp = torch.linalg.solve(A, Xtr.T @ Y)
    pred = (Xte @ Wp).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    In, Out, toks, pos = capture(rows)
    # labels: current token, previous token (within-seq), position bin
    T = 256
    cur = toks.copy()
    prev = np.full_like(toks, -1)
    prev[1:] = toks[:-1]; prev[pos == 0] = -1                      # previous token (undefined at pos 0)
    # restrict to TOPV frequent tokens for a bounded label set
    uniq, cnts = np.unique(toks, return_counts=True); topv = set(uniq[np.argsort(-cnts)[:TOPV]].tolist())
    remap = {t: i for i, t in enumerate(sorted(topv))}
    def to_lbl(arr): return np.array([remap.get(int(t), -1) for t in arr])
    cur_l = to_lbl(cur); prev_l = to_lbl(prev)
    posbin = (pos // 32).astype(np.int64); nposb = int(posbin.max())+1
    res = {}
    # current token
    v = cur_l >= 0
    res['current_from_input'] = round(decode_acc(In, cur_l, v, TOPV), 4)
    res['current_from_attn0out'] = round(decode_acc(Out, cur_l, v, TOPV), 4)
    # previous token
    vp = prev_l >= 0
    res['prev_from_input'] = round(decode_acc(In, prev_l, vp, TOPV), 4)
    res['prev_from_attn0out'] = round(decode_acc(Out, prev_l, vp, TOPV), 4)
    # shuffled null for prev-from-attn0out
    rng = np.random.RandomState(1); prev_sh = prev_l.copy(); prev_sh[vp] = rng.permutation(prev_sh[vp])
    res['prev_shuffled_null'] = round(decode_acc(Out, prev_sh, vp, TOPV), 4)
    # position
    res['pos_from_attn0out'] = round(decode_acc(Out, posbin, np.ones_like(posbin, bool), nposb), 4)
    prev_gain = res['prev_from_attn0out'] - res['prev_from_input']
    out = {'topv': TOPV, 'decode': res, 'prev_gain_out_over_in': round(prev_gain, 4),
           'pred_a_copysource': bool(res['prev_from_attn0out'] > res['prev_from_input'] + 0.1 and res['prev_from_attn0out'] > res['prev_shuffled_null'] + 0.1),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"decode top-1 acc (among {TOPV} tokens):", flush=True)
    print(f"  current: input {res['current_from_input']} | attn0-out {res['current_from_attn0out']}", flush=True)
    print(f"  PREVIOUS: input {res['prev_from_input']} | attn0-out {res['prev_from_attn0out']} | shuffled-null {res['prev_shuffled_null']}", flush=True)
    print(f"  position(bin): attn0-out {res['pos_from_attn0out']}", flush=True)
    print(f"(a) attn0 builds the copy-source (prev-token from output >> input & null): {out['pred_a_copysource']} (prev gain out-over-in {prev_gain:+.3f})", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
