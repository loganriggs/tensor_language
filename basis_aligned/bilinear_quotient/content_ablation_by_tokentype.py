"""Tie the content MECHANISM to its FUNCTION. The deep-middle content is a semantic topic manifold (§1055/§1064),
causally load-bearing (§1056) and topic-transporting (§1059). If it genuinely represents TOPIC, ablating it should hurt
the prediction of CONTENT/topical words (rare, meaning-bearing) far more than FUNCTION words (frequent, grammatical,
predictable from local syntax). Test: project the top-K content subspace OUT of the deep-middle residual stream (L6-14,
§1056 mechanism) and measure the per-target-token loss increase, BINNED BY TARGET FREQUENCY (rare = content/topical,
frequent = function). Control: ablate a random-K subspace (should be flat across frequency, or the reverse).

REGISTERED PREDICTIONS:
  (0) SANITY: no ablation = 0 loss increase; random-subspace ablation loss increase is much flatter across frequency.
  (a) CONTENT SUPPORTS TOPICAL PREDICTION: content-subspace ablation's loss increase is CONCENTRATED on LOW-FREQUENCY
      (content/topical) targets and small on HIGH-FREQUENCY (function) targets -> the content representation
      specifically supports content-word prediction (mechanism tied to function);
  (b) report per-frequency-decile loss increase for content vs random ablation; and the low-vs-high-freq ratio."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_ablation_by_tokentype_results.json'
NEVAL = 240; SEQ = 256; REF = [8, 10, 12]; ABL = list(range(6, 15)); K = 256; NBIN = 6
CAP = {}
PROJ = {'U': None}


def fwd_cap(idx):
    hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    out = fwd(idx, ablate=False)
    for h in hs: h.remove()
    return out


def fwd(idx, ablate=False):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ablate and PROJ['U'] is not None and li in ABL:
            U = PROJ['U']; x = x - (x @ U) @ U.T
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_token_ce(blocks, ablate):
    ces = []; tgts = []
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx, ablate).float(), -1); tf = tgt.reshape(-1)
        ce = -lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf]
        ces.append(ce); tgts.append(tf)
    return torch.cat(ces, 0), torch.cat(tgts, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    # content basis from pooled deep-middle content deviation
    for L in REF: CAP[L] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd_cap(idx)
    tok = torch.cat(idsL, 0); devsum = None
    for L in REF:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X; CAP[L] = []
    dev = devsum/len(REF); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); Ucontent = Vt[:K].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); Urand = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    # per-token CE
    PROJ['U'] = None; ce_base, tgt = per_token_ce(blocks, ablate=False)
    PROJ['U'] = Ucontent; ce_c, _ = per_token_ce(blocks, ablate=True)
    PROJ['U'] = Urand; ce_r, _ = per_token_ce(blocks, ablate=True); PROJ['U'] = None
    inc_c = (ce_c - ce_base).cpu().numpy(); inc_r = (ce_r - ce_base).cpu().numpy(); tgt = tgt.cpu().numpy()
    # target frequency over this corpus
    freq = np.bincount(tgt, minlength=V).astype(np.float64); logf = np.log(freq[tgt] + 1)
    # frequency-rank bins over TOKEN INSTANCES (equal-count deciles by log-freq)
    order = np.argsort(logf); binsz = len(order)//NBIN
    out = {'K': K, 'abl_range': [ABL[0], ABL[-1]], 'n_targets': int(len(tgt)), 'bins_low_to_high_freq': []}
    for b in range(NBIN):
        sel = order[b*binsz:(b+1)*binsz] if b < NBIN-1 else order[b*binsz:]
        out['bins_low_to_high_freq'].append({
            'mean_logfreq': round(float(logf[sel].mean()), 2),
            'content_ablation_loss_inc': round(float(inc_c[sel].mean()), 4),
            'random_ablation_loss_inc': round(float(inc_r[sel].mean()), 4)})
    lo = out['bins_low_to_high_freq'][0]['content_ablation_loss_inc']; hi = out['bins_low_to_high_freq'][-1]['content_ablation_loss_inc']
    out['content_low_over_high_ratio'] = round(lo/max(hi, 1e-6), 2)
    rlo = out['bins_low_to_high_freq'][0]['random_ablation_loss_inc']; rhi = out['bins_low_to_high_freq'][-1]['random_ablation_loss_inc']
    out['random_low_over_high_ratio'] = round(rlo/max(rhi, 1e-6), 2)
    out['pred_a_content_supports_topical'] = bool(lo > hi and out['content_low_over_high_ratio'] > out['random_low_over_high_ratio'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for b in out['bins_low_to_high_freq']:
        print(f"logfreq {b['mean_logfreq']}: content-abl +{b['content_ablation_loss_inc']} | random-abl +{b['random_ablation_loss_inc']}", flush=True)
    print(f"content low/high-freq ratio {out['content_low_over_high_ratio']} (random {out['random_low_over_high_ratio']}) | pred_a {out['pred_a_content_supports_topical']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
