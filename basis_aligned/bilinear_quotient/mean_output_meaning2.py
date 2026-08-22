"""[v2, embedding-fixed] WHAT DOES THE MODEL PREDICT WITH ZERO INPUT INFORMATION — is it the unigram/frequency prior? v1 (mean_output_meaning) was CONFOUNDED: the residual stream carries wte(idx), so "all components at their means" was NOT input-independent (pos-independence std 0.43). v2 ALSO replaces the token embedding with its global mean, giving a TRULY input-independent constant forward. If the model's constant pathway encodes the frequency prior, this fully-constant output should match the unigram (CE ~7.1, high correlation).
ORIGINAL: WHAT IS THE per-component CONSTANT OUTPUT (mean) that recovers 0.66-0.91 of each
component's benefit (§821)? Hypothesis: the components' constant outputs collectively build
the model's UNIGRAM / frequency 'default' prior — the prediction the model makes with no
token/position/context information, which is most of the achievable loss on uncertain tokens
(explaining why mean-only recovers so much). Test: run the model with EVERY component
replaced by its constant global-mean output; the resulting logits are input-INDEPENDENT (a
single constant distribution). Compare that constant distribution to the empirical unigram
frequency of the data.

REGISTERED PREDICTIONS:
  (0) SANITY: constant-only logits are ~identical across positions (input-independent);
  (a) FREQUENCY PRIOR: the constant-only output distribution matches the unigram frequency
      — its CE on real next-tokens ≈ the unigram-prior CE, and its probs correlate with
      unigram probs (Pearson r on log-probs >= 0.7) — so the per-component constants build
      the model's default/frequency baseline;
  (b) if NOT (CE far from unigram, low correlation), the constants encode something else
      (report what — e.g. a few dominant tokens);
  NULL: correlation with a SHUFFLED unigram distribution ~ 0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mean_output_meaning2_results.json'
NEVAL = 200
MEANS = {}; MODE = {'on': False}; EMB = {'mean': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if not MODE['on']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape
        v2 = MEANS[name].expand(y.reshape(-1, D).shape).reshape(sh).to(y.dtype)
        return (v2,) + tuple(o_[1:]) if isinstance(o_, tuple) else v2
    return hook


def forward_logits(idx):
    e = m.transformer.wte(idx)
    if MODE['on'] and EMB['mean'] is not None: e = EMB['mean'].expand_as(e)
    x = F.rms_norm(e, (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_means(rows):
    caps = {(w, L): [] for L in range(18) for w in ('attn', 'mlp')}
    hs = []
    for L in range(18):
        for w in ('attn', 'mlp'):
            def mkh(key):
                def h(mo, i_, o_): caps[key].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
                return h
            hs.append(comp(w, L).register_forward_hook(mkh((w, L))))
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
    for h in hs: h.remove()
    for k in caps: MEANS[k] = torch.cat(caps[k], 0).mean(0, keepdim=True)
    # global mean token embedding
    with torch.no_grad():
        es=[]
        for i in range(0, rows.shape[0], 4):
            idx = rows[i:i+4,:257].to(DEV)[:, :-1].contiguous(); es.append(m.transformer.wte(idx).detach().float().reshape(-1,D))
        EMB['mean']=torch.cat(es,0).mean(0).reshape(1,1,D)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    capture_means(rows)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in range(18) for w in ('attn', 'mlp')]
    # constant-only forward: logits input-independent; grab from a batch
    MODE['on'] = True
    idx = rows[:4, :257].to(DEV)[:, :-1].contiguous()
    logits = forward_logits(idx).float()          # (B,T,V)
    # position-independence check: std across positions of the argmax-token logit
    const_logit = logits.mean(dim=(0, 1))         # (V,) average constant logit
    pos_var = float(logits.reshape(-1, logits.shape[-1]).std(0).mean())
    MODE['on'] = False
    # empirical unigram from targets
    V = logits.shape[-1]
    tgt = rows[:, 1:257].reshape(-1).cpu().numpy()
    counts = np.bincount(tgt, minlength=V).astype(np.float64) + 1e-6
    unigram = counts / counts.sum()
    loguni = np.log(unigram)
    const_logp = F.log_softmax(const_logit, -1).cpu().numpy()
    # correlation (weight by unigram freq so rare-token noise doesn't dominate)
    wmask = counts > 5
    r = float(np.corrcoef(const_logp[wmask], loguni[wmask])[0, 1])
    rng = np.random.RandomState(0); sh = loguni[wmask].copy(); rng.shuffle(sh)
    r_null = float(np.corrcoef(const_logp[wmask], sh)[0, 1])
    # CE of constant-only predictor on real next tokens vs unigram-prior CE
    lp = F.log_softmax(const_logit, -1)
    tgt_t = torch.tensor(tgt, device=DEV)
    ce_const = float(F.nll_loss(lp.unsqueeze(0).expand(tgt_t.shape[0], -1), tgt_t))
    ce_unigram = float(-torch.tensor(np.log(unigram[tgt]), device=DEV).mean())
    # top tokens of the constant distribution
    top = torch.topk(const_logit, 10).indices.cpu().numpy().tolist()
    out = {'pos_independence_std': round(pos_var, 4), 'corr_const_vs_unigram': round(r, 4),
           'corr_const_vs_shuffled': round(r_null, 4), 'ce_constant_only': round(ce_const, 4),
           'ce_unigram_prior': round(ce_unigram, 4), 'ce_gap': round(ce_const - ce_unigram, 4),
           'top10_constant_tokens': top,
           'pred_a_frequency_prior': bool(r >= 0.7 and abs(ce_const - ce_unigram) < 0.5),
           'runtime_s': time.time()-t0}
    for h in hooks: h.remove()
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'constant-only: CE {ce_const:.3f} vs unigram-prior CE {ce_unigram:.3f} (gap {ce_const-ce_unigram:+.3f})', flush=True)
    print(f'corr(const, unigram) {r:.3f} | shuffled null {r_null:.3f} | pos-independence std {pos_var:.3f}', flush=True)
    print(f'(a) constants build the frequency prior: {out["pred_a_frequency_prior"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
