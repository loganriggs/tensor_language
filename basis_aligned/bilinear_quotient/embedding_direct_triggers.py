"""EMBEDDING DIRECT TRIGGERS -- trace the newline (635) and article
(636) triggers one step deeper: how much of each trigger is already in
the EMBEDDING->UNEMBEDDING direct path (a learned bigram table:
current-token embedding read straight by the unembedding), vs computed
by the 18 blocks?

The direct path skips every block: logits = unembed(rms_norm(embedding))
-- what the current token predicts about the next with zero context and
zero MLP computation, purely the learned embedding-unembedding bigram.
Comparing its trigger structure to the full model localizes how much of
each circuit is a 0-layer lookup vs assembled by the network.

REGISTERED PREDICTIONS:
  (0) SANITY: the direct path gives non-degenerate next-token
      distributions (overall P(newline), P(article) nonzero);
  (a) NEWLINE PARTLY EMBEDDING-LEVEL: the direct path shows a
      punctuation->newline elevation (P(newline) higher after . ! ? than
      after a word) -- the '.'-> newline bigram is partly in the
      embedding table -- but SMALLER than the full model's 28x jump (the
      blocks amplify it);
  (b) ARTICLE SPLIT PARTLY EMBEDDING-LEVEL: the direct path shows the
      be->a/an vs preposition->the split at least in sign, but weaker
      than the full model;
  (c) report direct vs full trigger elevations and the direct/full
      fraction for each;
  NULL: a non-trigger control (P(newline) after a random common word)
      shows no elevation in the direct path -- the embedding-level
      structure is specific to the trigger tokens."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'embedding_direct_triggers_results.json'
NFRESH = 48
NL1, NL2 = 198, 628
A_AN = [257, 281]
THE = [262, 383]
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
BE = {'is', 'was', 'are', 'were', 'be', 'been', 'being', 'am', "'s"}


def curcat(t):
    s = cl.d1(int(t)); st = s.strip().lower()
    if st in PREP:
        return 'prep'
    if st in BE:
        return 'be'
    if st and st[-1] in '.!?':
        return 'end_punct'
    return 'other'


@torch.no_grad()
def probs(fresh, direct):
    paa = torch.zeros(NFRESH, T); pth = torch.zeros(NFRESH, T)
    pnl = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        if not direct:
            x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        paa[i:i + B] = (p[..., A_AN[0]] + p[..., A_AN[1]]).cpu()
        pth[i:i + B] = (p[..., THE[0]] + p[..., THE[1]]).cpu()
        pnl[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
    return (paa.reshape(-1).numpy(), pth.reshape(-1).numpy(),
            pnl.reshape(-1).numpy())


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cur = fresh[:, :256].reshape(-1).numpy()
    cats = np.array([curcat(t) for t in cur])
    isword = np.array([cl.d1(int(t)).strip().isalpha() and curcat(t) == 'other'
                       for t in cur])
    endp = cats == 'end_punct'

    out = {}
    D_ = {}
    for name, direct in [('direct', True), ('full', False)]:
        paa, pth, pnl = probs(fresh, direct)
        nl_punct = float(pnl[endp].mean()); nl_word = float(pnl[isword].mean())
        be_pref = float(paa[cats == 'be'].mean() - pth[cats == 'be'].mean())
        prep_pref = float(paa[cats == 'prep'].mean() - pth[cats == 'prep'].mean())
        D_[name] = {'nl_punct': nl_punct, 'nl_word': nl_word,
                    'nl_elev': nl_punct - nl_word, 'be_pref': be_pref,
                    'prep_pref': prep_pref,
                    'overall_nl': float(pnl.mean()),
                    'overall_article': float((paa + pth).mean())}
        out[name] = {k: round(v, 5) for k, v in D_[name].items()}
        print(f'{name}: nl_punct {nl_punct:.4f} nl_word {nl_word:.4f} '
              f'(elev {nl_punct-nl_word:+.4f}); be_pref {be_pref:+.4f} '
              f'prep_pref {prep_pref:+.4f}', flush=True)

    nl_frac = D_['direct']['nl_elev'] / (D_['full']['nl_elev'] + 1e-9)
    p0 = D_['direct']['overall_nl'] > 0 and D_['direct']['overall_article'] > 0
    pa = D_['direct']['nl_elev'] > 0 and D_['direct']['nl_elev'] < D_['full']['nl_elev']
    pb = (D_['direct']['be_pref'] > D_['direct']['prep_pref'])
    null_ok = D_['direct']['nl_word'] < 0.5 * D_['direct']['nl_punct'] + 1e-6
    print(f'\n(0) direct non-degenerate: {p0}', flush=True)
    print(f'(a) newline elev direct {D_["direct"]["nl_elev"]:.4f} < full '
          f'{D_["full"]["nl_elev"]:.4f}, both>0: {pa} '
          f'(direct/full {nl_frac:.2f})', flush=True)
    print(f'(b) article split (be_pref>prep_pref) present in direct: {pb}',
          flush=True)
    print(f'NULL direct nl after word small vs punct: {null_ok}', flush=True)

    out.update({'newline_direct_fraction': round(float(nl_frac), 4),
                'pred_0': bool(p0), 'pred_a_newline_partly_emb': bool(pa),
                'pred_b_article_split_emb': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
