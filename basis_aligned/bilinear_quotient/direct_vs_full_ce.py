"""DIRECT VS FULL CE -- put a single number on the input-tracing phase
(634-639): how much of the model's predictive power is the 0-layer
embedding->unembedding BIGRAM, and how much is the 18 blocks' context
discrimination? And is the blocks' contribution concentrated on rare
(context-dependent) targets?

Direct path = unembed(rms_norm(embedding)), the context-blind bigram.
Full = all 18 blocks. Cross-entropy of each, overall and split by
frequent- vs rare-target, plus the uniform-guess reference log(V).

REGISTERED PREDICTIONS:
  (0) SANITY: the direct bigram beats a uniform guess by a wide margin
      (direct CE << log(V) ~ 10.8) -- the embedding bigram is a strong
      baseline;
  (a) BLOCKS HELP: full CE < direct CE overall -- the 18 blocks add real
      predictive power beyond the bigram;
  (b) CONTEXT HELPS RARE MORE: the blocks reduce CE by a larger absolute
      amount at rare-target positions than at frequent-target positions
      -- context discrimination matters most where the bigram is
      weakest;
  (c) report direct and full CE overall, frequent-target, rare-target,
      and the bigram's share of the model's information gain over
      uniform;
  NULL: the direct bigram's frequent-target CE is already low (frequent
      tokens are largely bigram-predictable) so the blocks help
      frequent targets little -- the gap is a rare-target phenomenon."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'direct_vs_full_ce_results.json'
NFRESH = 48
TOPK = 20


@torch.no_grad()
def ce_per_position(fresh, direct):
    V = m.lm_head.weight.shape[0]
    ces = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        if not direct:
            x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ce = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T)
        ces[i:i + B] = ce.cpu()
    return ces.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top for t in nxt])
    uniform = float(np.log(V))

    direct = ce_per_position(fresh, True)
    full = ce_per_position(fresh, False)

    def stats(ce):
        return {'all': float(ce.mean()), 'freq': float(ce[is_freq].mean()),
                'rare': float(ce[~is_freq].mean())}
    dS, fS = stats(direct), stats(full)
    # information gain over uniform, and bigram's share of it
    gain_bigram = uniform - dS['all']
    gain_full = uniform - fS['all']
    bigram_share = gain_bigram / (gain_full + 1e-9)
    help_freq = dS['freq'] - fS['freq']
    help_rare = dS['rare'] - fS['rare']

    print(f'uniform CE (log V) {uniform:.4f}', flush=True)
    print(f'direct: all {dS["all"]:.4f}  freq {dS["freq"]:.4f}  rare {dS["rare"]:.4f}',
          flush=True)
    print(f'full:   all {fS["all"]:.4f}  freq {fS["freq"]:.4f}  rare {fS["rare"]:.4f}',
          flush=True)
    print(f'blocks help: freq -{help_freq:.4f}  rare -{help_rare:.4f} nats', flush=True)
    print(f'bigram share of info gain over uniform: {bigram_share:.3f}', flush=True)

    p0 = dS['all'] < 0.6 * uniform
    pa = fS['all'] < dS['all']
    pb = help_rare > help_freq
    null_ok = dS['freq'] < dS['rare']
    print(f'\n(0) bigram beats uniform widely: {p0}', flush=True)
    print(f'(a) blocks help overall: {pa}', flush=True)
    print(f'(b) blocks help rare more than frequent: {pb}', flush=True)
    print(f'NULL bigram freq CE < rare CE: {null_ok}', flush=True)

    out = {'uniform_CE': round(uniform, 4), 'direct': {k: round(v, 4) for k, v in dS.items()},
           'full': {k: round(v, 4) for k, v in fS.items()},
           'help_freq_nats': round(help_freq, 4), 'help_rare_nats': round(help_rare, 4),
           'bigram_share_of_gain': round(bigram_share, 4),
           'pred_0': bool(p0), 'pred_a_blocks_help': bool(pa),
           'pred_b_rare_more': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
