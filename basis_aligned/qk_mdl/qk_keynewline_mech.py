"""KEY_newline head cluster mechanism characterization (the last open thread).
Cluster (census v2, top_predicate==KEY_newline): L0H8 L1H7 L2H4 L3H2 L9H8 L10H6 L11H4 L13H8 L16H4.
HISTORY: they attend newline-KEY positions in the census regression but PREDICT capital/punct
(attend != predict). Boundary-anchor hypothesis FALSIFIED (qk_newline_anchor.py: capital damage
was HIGHER not-post-newline). This script is a well-controlled characterization -- a clean negative
is a valid outcome.

Design: mean-ablation, in-distribution per-position zero point, held slice FW[448:600], paired
per-token dCE with SEs. Three mechanism tests:
 (0) attend-vs-predict: for each head, does the bilinear pattern actually place mass ON newline keys
     or AWAY (coefs are mixed-sign in the census); and which token-classes the cluster causally
     predicts (per-class dCE under cluster knockout).
 (a) DOCUMENT-STRUCTURE: split cluster-knockout damage by structured (dense-newline / list) vs prose
     context.
 (b) SEGMENT-RESET: split by absolute position (short vs long range) and by distance-since-last-
     newline (near vs far from the most recent reset).
 (c) NEWLINE-as-SINK / value: is the signal carried by the newline VALUE the head copies, or is the
     newline attention incidental? Contrast full head knockout vs (i) corrupt only newline-position
     VALUES (keep pattern) vs (ii) zero the pattern ON newline columns (keep values).

Forward pass copied VERBATIM from tier2_model.reference_forward / qk_bracket_patch.py (bilinear,
UNNORMALIZED pat=(s1*s2), v1 layer-0 value cache mixed via a.lamb, 30*tanh head).
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')

CLUSTER = [(0, 8), (1, 7), (2, 4), (3, 2), (9, 8), (10, 6), (11, 4), (13, 8), (16, 4)]
# held slice, 129 so idx=FW[:,:-1] is 128 wide
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[448:600][:, :129].to(DEV)
NSEQ, TT = FW.shape[0], FW.shape[1] - 1
CHUNK = 8

# --- token-class masks (verbatim classes from qk_keynewline_probe.py) ---
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which','you','have','he','they','has'}
MASKS = {k: torch.zeros(V, dtype=torch.bool) for k in ['subword','punct','capital','digit','newline','funcword']}
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if not lead and len(core) and core[0].isalpha() and core[0].islower(): MASKS['subword'][i] = True
    if len(core) and all(c in _P for c in core): MASKS['punct'][i] = True
    if lead and len(core) and core[0].isupper(): MASKS['capital'][i] = True
    if len(core) and all(c.isdigit() for c in core): MASKS['digit'][i] = True
    if 'Ċ' in s or '\n' in s: MASKS['newline'][i] = True
    if core.lower() in FUNC: MASKS['funcword'][i] = True
MASKS = {k: v.to(DEV) for k, v in MASKS.items()}
NLK = MASKS['newline']


@torch.no_grad()
def forward(idx, abl=None, ymean=None, vmean=None, want_means=False, want_diag=False):
    """abl: None
            | ('mean', heads)             -> per-position mean-ablate each head in `heads`
            | ('nlval', heads)            -> for each head, set its VALUE at newline-key columns to
                                             the per-position mean value (keep pattern)
            | ('nlpat', heads)            -> for each head, ZERO the pattern on newline-key columns
                                             (keep values)
            | ('allval', heads)           -> set value at ALL columns to mean (positive control)
    ymean/vmean: dict li-> per-position mean tensors (T,NH,HD).
    want_means: accumulate per-position sums of yh and v.
    want_diag: accumulate per-head bilinear-pattern-on-newline stats for CLUSTER heads.
    Forward core is VERBATIM qk_bracket_patch.py."""
    B, Tt = idx.shape
    nlk = NLK[idx]                                            # (B,T) key is a newline token
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'v': []} if want_means else None
    diag = {} if want_diag else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if want_means: acc['v'].append(v.sum(0))
        # --- value edits (before the einsum) ---
        if abl is not None and abl[0] in ('nlval', 'allval'):
            for (ll, hh) in abl[1]:
                if ll != li: continue
                vm = vmean[li][:, hh]                          # (T,HD) per-position mean value
                v = v.clone()
                if abl[0] == 'nlval':
                    sel = nlk                                  # only newline-key columns
                    v[:, :, hh] = torch.where(sel[..., None], vm[None].to(v.dtype), v[:, :, hh])
                else:
                    v[:, :, hh] = vm[None].to(v.dtype).expand(B, -1, -1)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        # --- pattern edit: zero newline columns for named heads ---
        if abl is not None and abl[0] == 'nlpat':
            for (ll, hh) in abl[1]:
                if ll != li: continue
                pat = pat.clone()
                pat[:, hh] = pat[:, hh].masked_fill(nlk[:, None, :].expand(B, Tt, Tt), 0.0)
        if want_diag:
            for hh in range(NH):
                if (li, hh) in CLUSTER:
                    P = pat[:, hh]                              # (B,T,T) unnormalized bilinear pattern
                    nlm = nlk[:, None, :].expand(B, Tt, Tt) & mask[None]
                    allm = mask[None].expand(B, Tt, Tt)
                    diag[(li, hh)] = {
                        'pat_nl_sum': float(P[nlm].sum()), 'pat_all_sum': float(P[allm].sum()),
                        'abs_nl_sum': float(P[nlm].abs().sum()), 'abs_all_sum': float(P[allm].abs().sum()),
                        'n_nl': int(nlm.sum()), 'n_all': int(allm.sum())}
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,T,NH,HD)
        if abl is not None and abl[0] == 'mean':
            yh = yh.clone()
            for (ll, hh) in abl[1]:
                if ll == li: yh[:, :, hh] = ymean[li][:, hh].unsqueeze(0)
        if want_means: acc['yh'].append(yh.sum(0))
        x = x + a.c_proj(yh.reshape(B, Tt, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    out = [logits]
    if want_means: out.append(acc)
    if want_diag: out.append(diag)
    return out[0] if len(out) == 1 else tuple(out)


# --- collect per-position means of yh and v over the held set ---
ysum = [torch.zeros(TT, NH, HD, device=DEV) for _ in range(NL)]
vsum = [torch.zeros(TT, NH, HD, device=DEV) for _ in range(NL)]
for i in range(0, NSEQ, CHUNK):
    _, acc = forward(FW[i:i+CHUNK, :-1], want_means=True)
    for li in range(NL):
        ysum[li] += acc['yh'][li]; vsum[li] += acc['v'][li]
YMEAN = {li: ysum[li] / NSEQ for li in range(NL)}
VMEAN = {li: vsum[li] / NSEQ for li in range(NL)}
print('per-position means collected', flush=True)

# --- attend-vs-predict diagnostic (one pass) ---
diag_agg = {lh: {'pat_nl_sum': 0.0, 'pat_all_sum': 0.0, 'abs_nl_sum': 0.0, 'abs_all_sum': 0.0, 'n_nl': 0, 'n_all': 0} for lh in CLUSTER}
for i in range(0, NSEQ, CHUNK):
    _, dg = forward(FW[i:i+CHUNK, :-1], want_diag=True)
    for lh in CLUSTER:
        for kk in diag_agg[lh]: diag_agg[lh][kk] += dg[lh][kk]
attend = {}
for lh in CLUSTER:
    d = diag_agg[lh]
    base_rate = d['n_nl'] / d['n_all']                          # newline share of key positions
    # signed mass fraction (>base => attends TO newline; <base => away/anti)
    signed_share = d['pat_nl_sum'] / d['pat_all_sum'] if d['pat_all_sum'] != 0 else float('nan')
    abs_share = d['abs_nl_sum'] / d['abs_all_sum'] if d['abs_all_sum'] != 0 else float('nan')
    mean_pat_nl = d['pat_nl_sum'] / d['n_nl']
    mean_pat_all = d['pat_all_sum'] / d['n_all']
    attend[f'L{lh[0]}H{lh[1]}'] = {
        'newline_keyrate': round(base_rate, 4),
        'abs_mass_share_on_newline': round(abs_share, 4),
        'signed_mass_share_on_newline': round(signed_share, 4),
        'mean_pat_at_newline': round(mean_pat_nl, 5),
        'mean_pat_overall': round(mean_pat_all, 5),
        'newline_vs_overall_ratio': round(mean_pat_nl / mean_pat_all, 3) if mean_pat_all != 0 else None}
print('ATTEND-ON-NEWLINE (abs_mass_share vs keyrate tells TO/AWAY):', flush=True)
for kk, vv in attend.items():
    print(f"  {kk}: keyrate {vv['newline_keyrate']}  abs_share {vv['abs_mass_share_on_newline']}  mean_pat nl/all {vv['mean_pat_at_newline']}/{vv['mean_pat_overall']} (x{vv['newline_vs_overall_ratio']})", flush=True)


# --- per-token CE collector for a given ablation over the whole held set ---
@torch.no_grad()
def collect_ce(abl):
    ces = []
    for i in range(0, NSEQ, CHUNK):
        idx = FW[i:i+CHUNK]
        lg = forward(idx[:, :-1], abl=abl, ymean=YMEAN, vmean=VMEAN)
        tgt = idx[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(idx.shape[0], -1)
        ces.append(ce.reshape(-1))
    return torch.cat(ces)                                       # (NSEQ*TT,)

# --- per-token classification masks (flattened over queries) ---
IDX = FW[:, :-1]; TGT = FW[:, 1:]
tgt_flat = TGT.reshape(-1)
capm = MASKS['capital'][tgt_flat]
punctm = MASKS['punct'][tgt_flat]
# context features from the QUERY-side stream idx (preceding tokens)
isnl = NLK[IDX].float()                                         # (B,T) current token is newline
# newline count in preceding window of 24 (structured/list = dense newlines)
win24 = torch.zeros_like(isnl)
for dd in range(1, 25): win24[:, dd:] += isnl[:, :-dd]
structured = (win24 >= 2).reshape(-1)
prose = (win24 == 0).reshape(-1)
# distance since most recent newline (segment-reset proxy)
pos = torch.arange(TT, device=DEV)[None].expand(FW.shape[0], -1)
last_nl = torch.full_like(pos, -1000)
for tstep in range(1, TT):
    prev = torch.where(isnl[:, tstep-1] > 0, torch.full((FW.shape[0],), tstep-1, device=DEV), last_nl[:, tstep-1])
    last_nl[:, tstep] = prev
dist_nl = (pos - last_nl)                                       # tokens since last newline
near_reset = ((dist_nl >= 1) & (dist_nl <= 6)).reshape(-1)
far_reset = (dist_nl > 20).reshape(-1)
# absolute position (short vs long range)
early = (pos < 32).reshape(-1)
late = (pos >= 96).reshape(-1)

def stat(d, msk):
    dm = d[msk]
    return {'n': int(msk.sum()), 'dCE': round(float(dm.mean()), 5), 'SE': round(float(dm.std() / np.sqrt(dm.numel())), 6)}

ce_clean = collect_ce(None)
ce_mean = collect_ce(('mean', CLUSTER))
d_mean = ce_mean - ce_clean
allq = torch.ones_like(capm)

res = {'cluster': [list(x) for x in CLUSTER], 'attend_on_newline': attend,
       'base_natural_CE': round(float(ce_clean.mean()), 4)}

# (0) predict: per-class dCE under full cluster knockout
res['cluster_knockout_by_class'] = {
    'all': stat(d_mean, allq), 'capital': stat(d_mean, capm), 'punct': stat(d_mean, punctm),
    'newline': stat(d_mean, MASKS['newline'][tgt_flat]), 'digit': stat(d_mean, MASKS['digit'][tgt_flat]),
    'subword': stat(d_mean, MASKS['subword'][tgt_flat])}

# (a) document-structure split (all tokens and capital-only)
res['a_structure_split'] = {
    'all_structured': stat(d_mean, structured), 'all_prose': stat(d_mean, prose),
    'capital_structured': stat(d_mean, capm & structured), 'capital_prose': stat(d_mean, capm & prose)}

# (b) segment-reset splits
res['b_reset_split'] = {
    'early_pos<32': stat(d_mean, early), 'late_pos>=96': stat(d_mean, late),
    'near_reset_1-6': stat(d_mean, near_reset), 'far_reset>20': stat(d_mean, far_reset),
    'capital_early': stat(d_mean, capm & early), 'capital_late': stat(d_mean, capm & late)}

# (c) newline-sink / value test
ce_nlval = collect_ce(('nlval', CLUSTER)); d_nlval = ce_nlval - ce_clean
ce_nlpat = collect_ce(('nlpat', CLUSTER)); d_nlpat = ce_nlpat - ce_clean
ce_allval = collect_ce(('allval', CLUSTER)); d_allval = ce_allval - ce_clean
res['c_sink_value'] = {
    'full_mean_knockout': {'all': stat(d_mean, allq), 'capital': stat(d_mean, capm)},
    'corrupt_newline_VALUES_only': {'all': stat(d_nlval, allq), 'capital': stat(d_nlval, capm)},
    'zero_pattern_ON_newline_cols': {'all': stat(d_nlpat, allq), 'capital': stat(d_nlpat, capm)},
    'corrupt_ALL_values_posctrl': {'all': stat(d_allval, allq), 'capital': stat(d_allval, capm)}}

print('\n--- (0) CLUSTER KNOCKOUT BY TARGET CLASS ---', flush=True)
for k, v in res['cluster_knockout_by_class'].items(): print(f"  {k}: +{v['dCE']} (SE {v['SE']}, n={v['n']})", flush=True)
print('\n--- (a) STRUCTURED vs PROSE ---', flush=True)
for k, v in res['a_structure_split'].items(): print(f"  {k}: +{v['dCE']} (SE {v['SE']}, n={v['n']})", flush=True)
print('\n--- (b) SEGMENT-RESET (position / dist-since-newline) ---', flush=True)
for k, v in res['b_reset_split'].items(): print(f"  {k}: +{v['dCE']} (SE {v['SE']}, n={v['n']})", flush=True)
print('\n--- (c) SINK / VALUE ---', flush=True)
for k, v in res['c_sink_value'].items(): print(f"  {k}: all +{v['all']['dCE']} (SE {v['all']['SE']}) | capital +{v['capital']['dCE']} (SE {v['capital']['SE']})", flush=True)

json.dump(res, open(f'{QK}/qk_keynewline_mech.json', 'w'), indent=2)
print('\nQK KEYNEWLINE MECH DONE', flush=True)
