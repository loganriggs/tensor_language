"""DIGIT CIRCUIT VERIFY 2 (corrected design; 732 lesson: to test whether a
behavior is a BIGRAM, keep the bigram head and ablate the REST, not the
bigram head itself). Item 9: digit CONTINUATION (prev digit -> next digit)
is a bigram carried by BLOCK-0 ATTENTION; digit INITIATION (first digit
after a non-digit) is COMPUTED by the later blocks. Test: keep block-0
attention active, ablate everything else (block-0 mlp + all of blocks
1-17). If continuation survives on the block-0 bigram while initiation
collapses, item 9 is confirmed. Uses MUCH more data (digit-continuation is
rare: ~41 per 40k tokens, so we run ~300k tokens).

REGISTERED PREDICTIONS:
  (0) SANITY: enough continuation positions (>=150) this time;
  (a) CONTINUATION IS A BLOCK-0 BIGRAM: with only block-0 attention kept,
      digit-CONTINUATION CE rises MUCH LESS than digit-INITIATION CE
      (initiation-rise / continuation-rise >= 2) -- continuation survives on
      the bigram, initiation needs the later blocks;
  (b) report full vs bigram-only CE for continuation and initiation;
  NULL: full-model continuation and initiation CE are similar (both
      predicted well) -- the difference is in what SURVIVES ablation."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digit_circuit_verify2_results.json'
NEVAL = 1200   # ~300k tokens (digit-continuation is rare)
KEEP = (0, 'attn')   # keep block-0 attention; ablate the rest
ABLATE = {'on': False}


def hook_factory(li, kind):
    def hook(mo, i_, o_):
        if not ABLATE['on']: return o_
        if (li, kind) == KEEP: return o_        # keep the bigram head
        return torch.zeros_like(o_)
    return hook


def is_digit_tok(t):
    s = cl.d1(int(t)).strip()
    return len(s) > 0 and s[0].isdigit()


@torch.no_grad()
def per_tok_ce(rows, n):
    ce = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0); lp = F.log_softmax(lg.float(), -1)
        ce.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(ce).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    ev = cl.fineweb_rows(NEVAL)
    cont = []; init = []
    for r in range(NEVAL):
        toks = ev[r, :257].tolist()
        for k in range(256):
            if is_digit_tok(toks[k+1]):
                (cont if is_digit_tok(toks[k]) else init).append(r*256 + k)
    cont = np.array(cont); init = np.array(init)
    print(f'continuation {len(cont)}, initiation {len(init)}', flush=True)

    hooks = []
    for li, blk in enumerate(m.transformer.h):
        hooks.append(blk.attn.c_proj.register_forward_hook(hook_factory(li, 'attn')))
        hooks.append(blk.mlp.register_forward_hook(hook_factory(li, 'mlp')))
    ABLATE['on'] = False; full = per_tok_ce(ev, NEVAL)
    ABLATE['on'] = True; bigram = per_tok_ce(ev, NEVAL); ABLATE['on'] = False
    for h in hooks: h.remove()

    def mean(a, idx): return float(a[idx].mean()) if len(idx) else float('nan')
    fc, fi = mean(full, cont), mean(full, init)
    bc, bi = mean(bigram, cont), mean(bigram, init)
    rc, ri = bc-fc, bi-fi
    ratio = ri / max(rc, 1e-6)
    print(f'continuation: full {fc:.3f}  bigram-only {bc:.3f}  rise {rc:+.3f}', flush=True)
    print(f'initiation:   full {fi:.3f}  bigram-only {bi:.3f}  rise {ri:+.3f}', flush=True)
    print(f'\ninitiation/continuation rise ratio: {ratio:.2f}', flush=True)

    p0 = len(cont) >= 150
    pa = ratio >= 2 and ri > 0.3
    print(f'(0) enough continuation positions: {p0}', flush=True)
    print(f'(a) continuation survives bigram >> initiation (ratio>=2): {pa}', flush=True)
    out = {'n_continuation': int(len(cont)), 'n_initiation': int(len(init)),
           'cont_full': round(fc,3), 'cont_bigram': round(bc,3), 'cont_rise': round(rc,3),
           'init_full': round(fi,3), 'init_bigram': round(bi,3), 'init_rise': round(ri,3),
           'ratio': round(ratio,3), 'pred_0': bool(p0), 'pred_a': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
