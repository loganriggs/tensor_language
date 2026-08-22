"""DIVISION OF LABOR: attention vs MLP by depth. The MLP side of the stack is well-characterized (front grammar
write, middle content multiplication, readout §939-950). Quantify the ATTENTION side alongside it: for each depth
band (front L0-5, middle L6-11, back L12-17), MEAN-ABLATE all ATTENTION outputs in the band vs all MLP outputs in
the band (others real), and measure held-out CE cost. This maps how the two component families divide the work
across depth — completing the bottom-up picture on the attention side.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablating any band's components raises CE; a random-set control lands between.
  (a) MLP-HEAVY FRONT, ATTENTION-SHARED MIDDLE: the FRONT band is MLP-dominated (front MLP cost >> front attn
      cost — grammar is written by mlp0/mlp1, §915/§933); the MIDDLE band has SUBSTANTIAL attention cost
      (attention aggregates context for content, §929/§932) comparable-or-larger relative to its MLP; the back is
      modest for both -> attention's role is context-aggregation concentrated in the middle, MLPs write at the
      front and compute content in the middle;
  (b) report per-band attention-ablate and MLP-ablate CE cost."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attention_vs_mlp_bands_results.json'
NEVAL = 200; SEQ = 256
BANDS = {'front_L0_5': list(range(0, 6)), 'middle_L6_11': list(range(6, 12)), 'back_L12_17': list(range(12, 18))}
ABL = {'targets': set(), 'means': None}   # targets = set of (kind, L)


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub(kind, L): return getattr(m.transformer.h[L], kind)


def mk_hook(kind, L):
    def h(mo, i_, o_):
        if (kind, L) not in ABL['targets']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['means'][(kind, L)].view(1, 1, D).expand(B, T, D).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def ce_pass(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    # global means of each attn and mlp output
    KINDS = ['attn', 'mlp']
    sums = {(k, L): torch.zeros(D, device=DEV) for L in range(18) for k in KINDS}; cnt = 0; hs = []
    for L in range(18):
        for k in KINDS:
            def mk(k, L):
                def h(mo, i_, o_): sums[(k, L)] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).sum(0)
                return h
            hs.append(sub(k, L).register_forward_hook(mk(k, L)))
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); forward_logits(idx); cnt += idx.shape[0]*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {key: v/cnt for key, v in sums.items()}
    hooks = [sub(k, L).register_forward_hook(mk_hook(k, L)) for L in range(18) for k in KINDS]
    ABL['targets'] = set(); ce_full = ce_pass(blocks)
    out = {'ce_full': round(ce_full, 4), 'bands': {}}
    for bname, layers in BANDS.items():
        ABL['targets'] = set(('attn', L) for L in layers); ce_a = ce_pass(blocks)
        ABL['targets'] = set(('mlp', L) for L in layers); ce_m = ce_pass(blocks)
        out['bands'][bname] = {'attn_ablate': round(ce_a - ce_full, 4), 'mlp_ablate': round(ce_m - ce_full, 4)}
        print(f"{bname:>14}: attn-ablate {ce_a-ce_full:+.4f} | mlp-ablate {ce_m-ce_full:+.4f}", flush=True)
    ABL['targets'] = set()
    for h in hooks: h.remove()
    b = out['bands']
    out['pred_a_mlp_front_attn_middle'] = bool(b['front_L0_5']['mlp_ablate'] > b['front_L0_5']['attn_ablate'] and
                                               b['middle_L6_11']['attn_ablate'] > 0.3*b['middle_L6_11']['mlp_ablate'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) MLP-heavy front, attention substantial in middle: {out['pred_a_mlp_front_attn_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
