"""Hypothesis-driven generalization tests (Logan): a functional claim must predict behavior on
inputs it was not discovered on. Pre-registered predictions:

H1 "the 45-component circuit DOES induction (content-agnostic match-and-copy)":
  (a) SHUFFLED prefixes (same tokens, order destroyed -> no natural n-grams; pure copy task):
      the circuit, found on natural prefixes, should retain ~90% of the (large) shuffled-induction
      advantage. Falsifier: retention collapses -> the circuit was natural-text-specific.
  (b) PERIODS 32 and 96 (discovered at 64): retention should hold. Falsifier: period-specific.
H2 "MLP0-3 is a CATEGORY engine (predicts next-token category, not identity)":
  Exact decomposition CE = categoryCE + withinCE (p_cat = prob mass on the target's 6-way category;
  categoryCE = -log p_cat; withinCE = -log p(tgt)/p_cat). Prediction: ablating MLP0-3 damages
  categoryCE disproportionately vs withinCE, and MORE so than a size-matched control ablation
  (MLP7-10). Falsifier: damage is category-neutral -> 'category engine' is the wrong name.
H3 "MLP1 feeds the two-branch match": on SHUFFLED prefixes too, ablating MLP1 alone should still
  destroy bilin18 induction (it serves the match, not natural-text content).
"""
import json, sys, ast
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
KEEP = set(ast.literal_eval(c) for c in json.load(open(f'{QK}/qk_induction_minimal.json'))['minimal_components'])

tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
CATM = torch.stack([CAT == c for c in range(6)])  # (6,V) bool


@torch.no_grad()
def run(EV, keep, MEAN, collect=False, ablate_mlp=frozenset()):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect: means[('m', li)] = mo.mean((0, 1))
        if (keep is not None and ('m', li) not in keep) or (li in ablate_mlp): mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(), means

P0 = 64
def make_eval(P, shuffled=False, n=48):
    pref = FINEWEB[:n, 1:1+P].clone()
    if shuffled:
        g = torch.Generator().manual_seed(11)
        for r in range(n): pref[r] = pref[r][torch.randperm(P, generator=g)]
    EV = torch.cat([pref, pref], 1).to(DEV)
    FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
    return EV, FIR, SEC

def adv_of(lg, EV, FIR, SEC):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

res = {'H1': {}, 'H2': {}, 'H3': {}}
print("== H1: circuit does content-agnostic copy ==", flush=True)
for name, P, sh in [('natural_P64', 64, False), ('shuffled_P64', 64, True), ('natural_P32', 32, False), ('natural_P96', 96, False)]:
    EV, FIR, SEC = make_eval(P, sh)
    _, MEAN = run(EV, None, None, True)
    af = adv_of(run(EV, None, None)[0], EV, FIR, SEC)
    an = adv_of(run(EV, set(), MEAN)[0], EV, FIR, SEC)
    am = adv_of(run(EV, KEEP, MEAN)[0], EV, FIR, SEC)
    ret = (am - an) / (af - an + 1e-9)
    res['H1'][name] = {'adv_full': round(af, 3), 'adv_none': round(an, 3), 'adv_circuit': round(am, 3), 'retention': round(ret, 3)}
    print(f"  {name}: full {af:+.3f} none {an:+.3f} circuit {am:+.3f} -> retention {ret:.1%}", flush=True)
    if name == 'shuffled_P64':
        a1 = adv_of(run(EV, set(ALL) - {('m', 1)}, MEAN)[0], EV, FIR, SEC)
        res['H3']['shuffled_ablate_MLP1'] = {'adv': round(a1, 3), 'vs_full': round(af, 3)}
        print(f"  H3 shuffled, ablate ONLY MLP1: adv {a1:+.3f} (full {af:+.3f})", flush=True)

print("== H2: MLP0-3 damages categoryCE, not withinCE ==", flush=True)
EVn = FINEWEB[:64, :128].to(DEV)
_, MEANn = run(EVn, None, None, True)
def cat_within(lg, EV):
    tgt = EV[:, 1:]; B, T = tgt.shape
    lp = F.log_softmax(lg.float(), -1)                     # (B,T,V)
    ls_cat = torch.stack([torch.logsumexp(lp[:, :, CATM[c]], -1) for c in range(6)], -1)  # (B,T,6)
    tc = CAT[tgt]                                          # target's category
    cat_ce = -ls_cat.gather(-1, tc.unsqueeze(-1)).squeeze(-1)             # -log p(category)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    within_ce = -(tok_lp + cat_ce)                                         # -log p(tok|cat)
    return cat_ce.mean().item(), within_ce.mean().item()
c0, w0 = cat_within(run(EVn, None, None)[0], EVn)
for name, abl in [('ablate_MLP0_3', {0, 1, 2, 3}), ('ablate_MLP7_10', {7, 8, 9, 10})]:
    keep = set(ALL) - {('m', l) for l in abl}
    c1, w1 = cat_within(run(EVn, keep, MEANn)[0], EVn)
    res['H2'][name] = {'d_categoryCE': round(c1 - c0, 4), 'd_withinCE': round(w1 - w0, 4),
                      'ratio_cat_over_within': round((c1 - c0) / max(w1 - w0, 1e-6), 2)}
    print(f"  {name}: d_categoryCE +{c1-c0:.4f} | d_withinCE +{w1-w0:.4f} | ratio {(c1-c0)/max(w1-w0,1e-6):.2f}", flush=True)
res['H2']['intact'] = {'categoryCE': round(c0, 4), 'withinCE': round(w0, 4)}
json.dump(res, open(f'{QK}/qk_hypothesis_tests.json', 'w'), indent=2)
print("QK HYPOTHESIS TESTS DONE", flush=True)
