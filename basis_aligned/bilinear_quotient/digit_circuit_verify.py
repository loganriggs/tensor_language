"""DIGIT CIRCUIT VERIFY (causal test of FINDINGS item 9: digit CONTINUATION
(prev token is a digit -> next digit) is a simple bigram, while digit
INITIATION (first digit after a non-digit, e.g. $ / page / space) is
COMPUTED by the blocks, 9.4x). Verify by comparing the DIRECT PATH (all 18
blocks ablated, embedding->unembedding only) to the full model at the two
position types. If item 9 holds, ablating the blocks should hurt INITIATION
far more than CONTINUATION (continuation survives on the bigram; initiation
needs the blocks).

REGISTERED PREDICTIONS:
  (0) SANITY: both position types non-empty; full-model digit CE reasonable;
  (a) COMPUTED INITIATION: ablating all blocks (direct path) raises digit-
      INITIATION CE much more than digit-CONTINUATION CE (ratio >= 2) -- the
      continuation is a bigram, the initiation is computed;
  (b) report full vs direct-path CE for continuation and initiation;
  NULL: for a control token-type with a simple bigram (e.g. continuation),
      the direct path is close to full."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digit_circuit_verify_results.json'
NEVAL = 160
ABLATE = {'on': False}


def hook(mo, i_, o_):
    return torch.zeros_like(o_) if ABLATE['on'] else o_


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
    # classify positions where NEXT token is a digit: continuation (cur is digit) vs initiation
    cont = []; init = []
    for r in range(NEVAL):
        toks = ev[r, :257].tolist()
        for k in range(256):
            if is_digit_tok(toks[k+1]):
                (cont if is_digit_tok(toks[k]) else init).append(r*256 + k)
    cont = np.array(cont); init = np.array(init)
    print(f'digit-continuation positions {len(cont)}, digit-initiation {len(init)}', flush=True)

    hooks = []
    for blk in m.transformer.h:
        hooks.append(blk.attn.c_proj.register_forward_hook(hook))
        hooks.append(blk.mlp.register_forward_hook(hook))
    # note: zeroing attn.c_proj + mlp output leaves the residual = embedding + lambda re-injections
    ABLATE['on'] = False; full = per_tok_ce(ev, NEVAL)
    ABLATE['on'] = True; direct = per_tok_ce(ev, NEVAL); ABLATE['on'] = False
    for h in hooks: h.remove()

    def mean(a, idx): return float(a[idx].mean()) if len(idx) else float('nan')
    fc, fi = mean(full, cont), mean(full, init)
    dc, di = mean(direct, cont), mean(direct, init)
    rise_c = dc - fc; rise_i = di - fi
    ratio = rise_i / max(rise_c, 1e-6)
    print(f'continuation: full CE {fc:.3f}  direct-path CE {dc:.3f}  rise {rise_c:+.3f}', flush=True)
    print(f'initiation:   full CE {fi:.3f}  direct-path CE {di:.3f}  rise {rise_i:+.3f}', flush=True)
    print(f'\ninitiation/continuation blocks-needed ratio: {ratio:.2f}', flush=True)

    p0 = len(cont) > 20 and len(init) > 20
    pa = ratio >= 2 and rise_i > 0.3
    print(f'(a) initiation computed >> continuation (ratio>=2): {pa}', flush=True)

    out = {'n_continuation': int(len(cont)), 'n_initiation': int(len(init)),
           'cont_full_ce': round(fc,3), 'cont_direct_ce': round(dc,3), 'cont_rise': round(rise_c,3),
           'init_full_ce': round(fi,3), 'init_direct_ce': round(di,3), 'init_rise': round(rise_i,3),
           'ratio': round(ratio,3), 'pred_0': bool(p0), 'pred_a_computed_initiation': bool(pa),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
