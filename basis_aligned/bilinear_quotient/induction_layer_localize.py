"""WHICH specific layers host INDUCTION? §952 localized induction to front attention (L0-5, mean-ablate costs
inductable positions +5.2, ~= all attention). Pinpoint it: mean-ablate EACH attention layer individually and
measure the CE cost on INDUCTABLE positions (current token seen earlier with the same next token). Also report the
first_mention cost per layer for contrast. This finds the induction layer(s).

REGISTERED PREDICTIONS:
  (0) SANITY: summed single-layer inductable costs are in the ballpark of the front-band cost (§952 +5.2, allowing
      super/sub-additivity); late layers ~0 on induction.
  (a) INDUCTION IS LOCALIZED to 1-3 specific FRONT layers (not uniform across L0-5): a small number of early
      attention layers carry most of the inductable-position cost, with a clear peak; later layers ~0 ->
      induction is a localized early circuit;
  (b) report per-layer attention mean-ablate Δce on inductable and first_mention positions."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_layer_localize_results.json'
NEVAL = 200; SEQ = 256
ABL = {'L': -1, 'means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def attn_hook(L):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['means'][L].view(1, 1, D).expand(B, T, D).clone()
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def per_pos_ce(blocks):
    outs = []
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        lpf = lp.reshape(-1, lp.shape[-1])
        outs.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    inductable = np.zeros((nb, SEQ-1), bool); firstment = np.zeros((nb, SEQ-1), bool)
    for r in range(nb):
        seen = set(); big = {}
        for p in range(SEQ-1):
            cur = int(S[r, p]); nx = int(S[r, p+1]); firstment[r, p] = nx not in seen
            if cur in big and big[cur] == nx: inductable[r, p] = True
            big[cur] = nx; seen.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable
    sums = {L: torch.zeros(D, device=DEV) for L in range(18)}; cnt = 0; hs = []
    for L in range(18):
        def mk(L):
            def h(mo, i_, o_): sums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).sum(0)
            return h
        hs.append(m.transformer.h[L].attn.register_forward_hook(mk(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous()); cnt += min(8, nb-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {L: sums[L]/cnt for L in range(18)}
    hooks = [m.transformer.h[L].attn.register_forward_hook(attn_hook(L)) for L in range(18)]
    ABL['L'] = -1; base = per_pos_ce(blocks)
    out = {'baseline_inductable': round(float(base[inductable].mean()), 4),
           'baseline_first_mention': round(float(base[firstment].mean()), 4), 'per_layer': {}}
    for L in range(18):
        ABL['L'] = L; w = per_pos_ce(blocks); ABL['L'] = -1; delta = w - base
        out['per_layer'][str(L)] = {'inductable': round(float(delta[inductable].mean()), 4),
                                    'first_mention': round(float(delta[firstment].mean()), 4)}
        print(f"L{L:>2} attn-ablate: inductable {out['per_layer'][str(L)]['inductable']:+.4f} | first_mention {out['per_layer'][str(L)]['first_mention']:+.4f}", flush=True)
    for h in hooks: h.remove()
    ind = {L: out['per_layer'][str(L)]['inductable'] for L in range(18)}
    top = sorted(ind, key=ind.get, reverse=True)[:3]
    out['top3_induction_layers'] = top; out['top3_share'] = round(sum(ind[L] for L in top)/max(sum(max(v,0) for v in ind.values()), 1e-6), 3)
    out['pred_a_induction_localized'] = bool(out['top3_share'] > 0.6)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top-3 induction layers {top} carry {out['top3_share']:.2f} of total inductable-ablation cost", flush=True)
    print(f"(a) induction localized to a few layers: {out['pred_a_induction_localized']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
