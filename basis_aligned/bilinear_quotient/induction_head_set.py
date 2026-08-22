"""CAUSALLY VALIDATE the identified induction heads (§954: L2h5, L5h5, L8h3, L3h8, L8h4) as a COOPERATIVE SET.
Each alone is tiny (top +0.12); do they act super-additively as a set? Mean-ablate the TOP-5 induction heads
TOGETHER and measure the CE cost on INDUCTABLE positions, vs (i) the sum of their individual costs (0.37, §954),
(ii) 5 RANDOM heads ablated together (control), (iii) the whole front-attention band (§952 +5.21, upper anchor).

REGISTERED PREDICTIONS:
  (0) SANITY: 5 random heads together cost little on inductable; baseline inductable ~0.67.
  (a) COOPERATIVE INDUCTION SET: ablating the top-5 induction heads together costs inductable MUCH more than the
      sum of their singles (0.37) AND much more than 5 random heads -> they form a genuine cooperative induction
      head-set (super-additive); but well below the full front band (5.21), since induction is broader still;
  (b) report top-5-together vs sum-of-singles vs 5-random vs front-band, on inductable and (contrast) first_mention."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; HEAD = 128; NH = 9; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_head_set_results.json'
NEVAL = 160; SEQ = 256
TOP5 = [(2, 5), (5, 5), (8, 3), (3, 8), (8, 4)]
RAND5 = [(1, 2), (4, 7), (6, 0), (7, 6), (0, 3)]
ABL = {'heads': set(), 'band_attn': set(), 'means': None, 'attn_means': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def cproj_prehook(L):
    def h(mo, args):
        act = [hh for (ll, hh) in ABL['heads'] if ll == L]
        if not act: return args
        x = args[0].clone()
        for hh in act: x[:, :, hh*HEAD:(hh+1)*HEAD] = ABL['means'][(L, hh)].view(1, 1, HEAD)
        return (x,) + tuple(args[1:])
    return h


def attn_hook(L):
    def h(mo, i_, o_):
        if L not in ABL['band_attn']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = ABL['attn_means'][L].view(1, 1, D).expand(B, T, D).clone()
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
        outs.append((-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
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
    Ls = sorted(set([l for l, _ in TOP5+RAND5]))
    # head means (c_proj input) + attn means (front band)
    hsums = {(l, h): torch.zeros(HEAD, device=DEV) for (l, h) in TOP5+RAND5}
    asums = {L: torch.zeros(D, device=DEV) for L in range(6)}; cnt = 0; hs = []
    for L in Ls:
        def mkp(L):
            def hook(mo, args):
                x = args[0].detach().float().reshape(-1, D)
                for (ll, hh) in TOP5+RAND5:
                    if ll == L: hsums[(ll, hh)] += x[:, hh*HEAD:(hh+1)*HEAD].sum(0)
                return None
            return hook
        hs.append(m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mkp(L)))
    for L in range(6):
        def mka(L):
            def hook(mo, i_, o_): asums[L] += (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D).sum(0)
            return hook
        hs.append(m.transformer.h[L].attn.register_forward_hook(mka(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous()); cnt += min(8, nb-i)*(SEQ-1)
    for h in hs: h.remove()
    ABL['means'] = {k: v/cnt for k, v in hsums.items()}; ABL['attn_means'] = {L: asums[L]/cnt for L in range(6)}
    phooks = [m.transformer.h[L].attn.c_proj.register_forward_pre_hook(cproj_prehook(L)) for L in range(9)]
    ahooks = [m.transformer.h[L].attn.register_forward_hook(attn_hook(L)) for L in range(6)]
    ABL['heads'] = set(); ABL['band_attn'] = set(); base = per_pos_ce(blocks)
    def cost(w): return {'inductable': round(float((w[inductable]-base[inductable]).mean()), 4),
                         'first_mention': round(float((w[firstment]-base[firstment]).mean()), 4)}
    out = {'baseline_inductable': round(float(base[inductable].mean()), 4), 'conditions': {}}
    ABL['heads'] = set(TOP5); w = per_pos_ce(blocks); out['conditions']['top5_together'] = cost(w); ABL['heads'] = set()
    ABL['heads'] = set(RAND5); w = per_pos_ce(blocks); out['conditions']['rand5_together'] = cost(w); ABL['heads'] = set()
    ABL['band_attn'] = set(range(6)); w = per_pos_ce(blocks); out['conditions']['front_band'] = cost(w); ABL['band_attn'] = set()
    for hh in phooks+ahooks: hh.remove()
    t5 = out['conditions']['top5_together']['inductable']; r5 = out['conditions']['rand5_together']['inductable']
    out['sum_of_top5_singles_ref'] = 0.37
    out['pred_a_cooperative_set'] = bool(t5 > 0.37 and t5 > 3*max(r5, 1e-6))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top5-together {out['conditions']['top5_together']} | rand5 {out['conditions']['rand5_together']} | front-band {out['conditions']['front_band']}", flush=True)
    print(f"top5-together inductable {t5} vs sum-of-singles 0.37 vs rand5 {r5}", flush=True)
    print(f"(a) cooperative induction head-set: {out['pred_a_cooperative_set']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
