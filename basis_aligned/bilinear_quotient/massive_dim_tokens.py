"""MASSIVE DIM TOKENS -- what drives the massive activations (676-677)?
In LLMs, massive activations often concentrate on a few "sink" tokens
(the first token, delimiters, newlines) regardless of content. Test
whether this model's massive-dim activations spike on specific structural
tokens/positions, or are spread across content.

For the top persistent massive dims (measured on the block-17 residual),
look at |activation| by: (1) sequence position (is position 0 / early
special?); (2) current-token type (newline, punct, space_word, etc.).
Report where the massive activations concentrate.

REGISTERED PREDICTIONS:
  (0) SANITY: the chosen dims are massive (RMS >> median);
  (a) SINK PATTERN: the massive-dim magnitude is concentrated -- either
      on the first position(s) or on specific structural tokens
      (newline/punct) far above the average token -- rather than uniform;
  (b) report the top dims' mean |activation| by position bucket (0, 1-4,
      5-32, 33+) and by token class;
  NULL: position 0 vs mid-sequence magnitude differ (if a positional
      sink) OR a token class stands out (if a token sink) -- report which."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'massive_dim_tokens_results.json'
NFRESH = 48
LJ = 17

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as'}


def classify(s):
    if chr(10) in s:
        return 'newline'
    t = s.strip().lower()
    if not t:
        return 'space'
    if t in DET or t in PREP:
        return 'function'
    if t[0].isdigit():
        return 'digit'
    if all(not c.isalnum() for c in t):
        return 'punct'
    if s.strip()[:1].isupper():
        return 'capitalized'
    if s.startswith(' '):
        return 'space_word'
    return 'subword'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)

    # capture residual after block 17 (B,T,D) with position preserved
    acts = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        acts.append(x.detach().float().cpu())          # (B,T,D)
    X = torch.cat(acts, 0).numpy()                     # (NFRESH,T,D)
    rms = np.sqrt((X.reshape(-1, D) ** 2).mean(0))
    med = float(np.median(rms))
    top_dims = np.argsort(-rms)[:5].tolist()
    print(f'top dims {top_dims} RMS {[round(float(rms[d]),0) for d in top_dims]} '
          f'(median {med:.0f})', flush=True)

    absX = np.abs(X[:, :, top_dims]).mean(2)           # (NFRESH,T) mean |act| over top dims
    cur = fresh[:, :256].numpy()

    # by position bucket
    pos = np.arange(T)
    buckets = {'pos0': pos == 0, 'pos1-4': (pos >= 1) & (pos <= 4),
               'pos5-32': (pos >= 5) & (pos <= 32), 'pos33+': pos >= 33}
    by_pos = {}
    for name, mk in buckets.items():
        by_pos[name] = round(float(absX[:, mk].mean()), 1)
    print('by position:', by_pos, flush=True)

    # by current-token class
    cls = np.array([[classify(cl.d1(int(cur[r, j]))) for j in range(T)]
                    for r in range(NFRESH)])
    by_cls = {}
    for c in ['newline', 'punct', 'digit', 'capitalized', 'space_word', 'subword',
              'function', 'space']:
        mk = cls == c
        if mk.sum() >= 20:
            by_cls[c] = round(float(absX[mk].mean()), 1)
    overall = round(float(absX.mean()), 1)
    print(f'overall mean |act| on top dims: {overall}', flush=True)
    print('by token class:', dict(sorted(by_cls.items(), key=lambda kv: -kv[1])), flush=True)

    p0 = float(rms[top_dims[0]]) > 5 * med
    pos_sink = by_pos['pos0'] > 2 * by_pos['pos33+'] or by_pos['pos1-4'] > 2 * by_pos['pos33+']
    tok_top = max(by_cls, key=by_cls.get)
    tok_sink = by_cls[tok_top] > 2 * overall
    pa = pos_sink or tok_sink
    print(f'\n(0) massive dims: {p0}', flush=True)
    print(f'(a) sink pattern (positional or token): {pa} '
          f'(pos0 {by_pos["pos0"]} vs pos33+ {by_pos["pos33+"]}; '
          f'top class {tok_top} {by_cls[tok_top]} vs overall {overall})', flush=True)

    out = {'top_dims': top_dims, 'top_rms': [round(float(rms[d]), 1) for d in top_dims],
           'median_rms': med, 'by_position': by_pos, 'overall': overall,
           'by_token_class': by_cls, 'positional_sink': bool(pos_sink),
           'token_sink': bool(tok_sink), 'top_class': tok_top,
           'pred_0': bool(p0), 'pred_a_sink': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
