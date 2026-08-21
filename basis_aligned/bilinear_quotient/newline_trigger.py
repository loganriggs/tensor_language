"""NEWLINE TRIGGER -- trace the newline context signal (634: newline is
the lone context-driven class, carried by front attention) to concrete
INPUT features, and confirm front attention carries them.

634 found P(newline) is written by front ATTENTION (context), not the
token-local MLP (which suppresses it). What context? Two candidate input
features: (i) the current token is sentence-ending punctuation ('.', '!',
'?') -- a bigram trigger; (ii) line length -- tokens since the last
newline (a positional / how-long-this-line-is signal that rotary
attention can carry). This measures P(newline) grouped by each feature,
at baseline vs with front [0-2] attention ablated vs front MLP ablated.

REGISTERED PREDICTIONS:
  (0) SANITY: overall mean P(newline) is a few percent (matches prior
      runs ~0.02-0.03);
  (a) PUNCT TRIGGER: P(newline) is markedly higher when the current
      token is sentence-ending punctuation than when it is a mid-line
      word;
  (b) LENGTH TREND: P(newline) rises with line length (tokens since the
      last newline);
  (c) ATTENTION CARRIES IT: ablating front ATTENTION reduces/flattens
      both the punctuation elevation and the length trend, far more than
      ablating the front MLP does (634: the MLP suppresses newline, so
      MLP ablation should not flatten the context signal);
  NULL: front-MLP ablation does not flatten the punct/length structure
      the way front-attention ablation does -- the trigger is
      attention-carried."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_trigger_results.json'
NFRESH = 48
NL1, NL2 = 198, 628
LEN_BUCKETS = [(0, 10), (10, 30), (30, 60), (60, 300)]


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def pnewline(fresh, blocks, kind):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill_hook))
    p_all = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        p_all[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
    for h in handles:
        h.remove()
    return p_all.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    idx = fresh[:, :256]                       # current tokens (predict next)

    # feature 1: current token is sentence-ending punctuation
    def is_end_punct(t):
        s = cl.d1(int(t)).strip()
        return len(s) > 0 and s[-1] in '.!?'
    cur = idx.reshape(-1).numpy()
    endp = np.array([is_end_punct(t) for t in cur])
    isword = np.array([cl.d1(int(t)).strip().isalpha() for t in cur])

    # feature 2: line length = tokens since last newline, per position
    linelen = np.zeros((NFRESH, 256), dtype=int)
    for r in range(NFRESH):
        c = 0
        for j in range(256):
            linelen[r, j] = c
            c = 0 if chr(10) in cl.d1(int(idx[r, j])) else c + 1
    linelen = linelen.reshape(-1)

    conds = {'baseline': (None, None), 'front_attn_abl': ([0, 1, 2], 'attn'),
             'front_mlp_abl': ([0, 1, 2], 'mlp')}
    P = {name: pnewline(fresh, b, k) for name, (b, k) in conds.items()}

    out = {'overall': {}, 'by_punct': {}, 'by_length': {}}
    for name, p in P.items():
        out['overall'][name] = round(float(p.mean()), 5)
        out['by_punct'][name] = {
            'end_punct': round(float(p[endp].mean()), 5),
            'word': round(float(p[isword].mean()), 5),
            'elevation': round(float(p[endp].mean() - p[isword].mean()), 5)}
        out['by_length'][name] = [round(float(p[(linelen >= lo) & (linelen < hi)].mean()), 5)
                                  for lo, hi in LEN_BUCKETS]
        print(f'{name:15s} overall {out["overall"][name]:.5f}  '
              f'end-punct {out["by_punct"][name]["end_punct"]:.5f}  '
              f'word {out["by_punct"][name]["word"]:.5f}  '
              f'by-len {out["by_length"][name]}', flush=True)

    base = out['by_punct']['baseline']
    p0 = 0.005 < out['overall']['baseline'] < 0.1
    pa = base['end_punct'] > 1.5 * base['word']
    bl = out['by_length']['baseline']
    pb = bl[-1] > bl[0]
    # (c) attention ablation flattens elevation more than mlp ablation
    elev_base = base['elevation']
    elev_attn = out['by_punct']['front_attn_abl']['elevation']
    elev_mlp = out['by_punct']['front_mlp_abl']['elevation']
    pc = (elev_base - elev_attn) > (elev_base - elev_mlp)
    print(f'\n(0) overall plausible: {p0}', flush=True)
    print(f'(a) end-punct elevates newline (>1.5x word): {pa} '
          f'({base["end_punct"]:.4f} vs {base["word"]:.4f})', flush=True)
    print(f'(b) rises with line length: {pb} ({bl})', flush=True)
    print(f'(c) attn ablation flattens elevation more than mlp: {pc} '
          f'(base {elev_base:.4f} -> attn {elev_attn:.4f}, mlp {elev_mlp:.4f})',
          flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_punct': bool(pa),
                'pred_b_length': bool(pb), 'pred_c_attn_carries': bool(pc),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
