"""ECG causal feature circuit (Logan's directive): treat each block-0 MLP inner
activation space (192 units) as the feature layer. For each capable code define a
FEATURE DIRECTION (pos-minus-neg mean inner activation), then CAUSALLY insert/remove
that feature on the HELD-OUT TEST SET and measure:
  (1) target-code prob change (remove -> drop on positives; insert -> rise on negatives)
  (2) off-target code changes -> which diagnoses SHARE the feature (correlation via shared circuit)
  (3) AUC collapse when the feature is projected out (is it causally necessary/sufficient?)
Also render the EXACT waveform each feature reads: top-activating real test patches,
raw 12-lead x 50-sample average = the empirical morphology template.
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV)


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, edit=None, return_inner0=False):
    """edit: fn(inner (B,T,INNER)) -> inner applied at block-0 MLP (li==0)."""
    xn = norm(x)
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    inner0 = None
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
        if li == 0:
            if return_inner0:
                inner0 = inner
            if edit is not None:
                inner = edit(inner)
        h = h + (inner @ W[mw+'Dn.weight'].T)
    logit = F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']
    return (logit, inner0) if return_inner0 else logit


@torch.no_grad()
def run(edit=None):
    outs = []
    for i in range(0, len(Xte), 2048):
        outs.append(forward(Xte[i:i+2048], edit).float())
    return torch.cat(outs)


def auc_col(s, c):
    lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0:
        return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum() - p*(p+1)/2) / (p*n))


# ---- capture block-0 inner activations (pos-pooled per ECG) on the test set ----
inner_pool = []
base_logit = []
for i in range(0, len(Xte), 2048):
    lg, inn = forward(Xte[i:i+2048], return_inner0=True)
    base_logit.append(lg.float()); inner_pool.append(inn.mean(1).float())   # mean over 20 positions
base_logit = torch.cat(base_logit); INN = torch.cat(inner_pool)            # (Nte, INNER)
base = np.array([auc_col(base_logit, c) for c in range(NCLS)])
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
print(f'{len(capable)} capable codes; INN {tuple(INN.shape)}', flush=True)

# ---- feature direction per code: mean(inner|pos) - mean(inner|neg) ----
feat = torch.zeros(NCLS, INNER, device=DEV)
for c in capable:
    m = Yte[:, c].bool()
    feat[c] = INN[m].mean(0) - INN[~m].mean(0)
featn = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-8)

# ---- CAUSAL REMOVE: project the code's feature direction out of every position's inner ----
def make_remove(c):
    u = featn[c]
    def edit(inner):                                   # inner (B,T,INNER)
        proj = (inner @ u).unsqueeze(-1) * u           # component along feature
        return inner - proj
    return edit

# ---- CAUSAL INSERT: add alpha * feature (scaled to typical activation) to every position ----
def make_insert(c, alpha):
    v = feat[c]
    def edit(inner):
        return inner + alpha * v
    return edit


# baseline mean prob on test positives/negatives (for target)
def probs(s):
    return torch.sigmoid(s)

steer = {}
# For a compact but rigorous readout: measure REMOVE effect (AUC + mean-prob on positives)
# and its OFF-TARGET spillover to other codes -> shared-feature correlation.
offtarget = np.zeros((len(capable), len(capable)))     # remove feature of row-code, watch col-code AUC drop
for ri, c in enumerate(capable):
    s_rm = run(make_remove(c))
    p_base = base_logit  # reuse
    # target
    tgt_auc_rm = auc_col(s_rm, c)
    mpos_base = float(probs(base_logit[:, c])[Yte[:, c].bool()].mean())
    mpos_rm = float(probs(s_rm[:, c])[Yte[:, c].bool()].mean())
    # insert on negatives (alpha=2 => 2x the mean pos-neg gap)
    s_in = run(make_insert(c, 2.0))
    mneg_base = float(probs(base_logit[:, c])[~Yte[:, c].bool()].mean())
    mneg_in = float(probs(s_in[:, c])[~Yte[:, c].bool()].mean())
    # off-target AUC change from REMOVE (how much removing c's feature moves other codes)
    for ci, c2 in enumerate(capable):
        offtarget[ri, ci] = base[c2] - auc_col(s_rm, c2)
    steer[CODES[c]] = {
        'auc': round(float(base[c]), 3),
        'auc_after_remove': round(tgt_auc_rm, 3),
        'auc_drop_remove': round(float(base[c] - tgt_auc_rm), 3),
        'meanprob_pos_base': round(mpos_base, 3), 'meanprob_pos_after_remove': round(mpos_rm, 3),
        'meanprob_neg_base': round(mneg_base, 3), 'meanprob_neg_after_insert': round(mneg_in, 3),
    }
    if ri % 5 == 0:
        print(f'  steered {ri}/{len(capable)} ({CODES[c]}): remove drop {base[c]-tgt_auc_rm:.3f}, '
              f'insert neg {mneg_base:.3f}->{mneg_in:.3f}', flush=True)

# ---- shared-feature correlated diagnoses: top off-target pairs ----
np.fill_diagonal(offtarget, 0.0)
pairs = []
for ri in range(len(capable)):
    for ci in range(len(capable)):
        if offtarget[ri, ci] > 0.02:
            pairs.append((CODES[capable[ri]], CODES[capable[ci]], round(float(offtarget[ri, ci]), 3)))
pairs.sort(key=lambda t: -t[2])

# ---- render exact feature waveform: top-activating test patches per code's top unit ----
# reconstruct per-position inner (not pooled) for a subset to find top patches
def top_unit(c):
    return int(torch.argmax(feat[c].abs()))
render = {}
Xn_all = norm(Xte)                                   # (Nte,12,1000)
# recompute per-position inner for all test (one pass, keep on gpu in chunks)
inner_pp = []
for i in range(0, len(Xte), 2048):
    _, inn = forward(Xte[i:i+2048], return_inner0=True)
    inner_pp.append(inn.float())
inner_pp = torch.cat(inner_pp)                        # (Nte, NP, INNER)
for c in capable:
    u = top_unit(c)
    act = inner_pp[:, :, u]                           # (Nte, NP) activation of unit u at each time-patch
    flat = act.reshape(-1)
    k = min(200, flat.numel())
    topv, topi = torch.topk(flat, k)
    ex = topi // NP; pos = topi % NP                  # which ecg, which time-patch
    # gather the raw (normalized) 12-lead x 50-sample waveform of those patches
    tmpl = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        tmpl += Xn_all[e, :, p*PT:(p+1)*PT]
    tmpl /= k
    render[CODES[c]] = {'top_unit': u, 'mean_topact': round(float(topv.mean()), 3),
                        'template_leads_peak': {LEADS[L]: round(float(tmpl[L].abs().max()), 2) for L in range(NLEAD)}}
np.save(f'{QK}/ecg_feature_dirs.npy', feat.cpu().numpy())
np.save(f'{QK}/ecg_offtarget.npy', offtarget)

res = {'n_capable': len(capable),
       'mean_auc_drop_remove': round(float(np.mean([steer[CODES[c]]['auc_drop_remove'] for c in capable])), 3),
       'codes_remove_collapses_auc>=0.1': [CODES[c] for c in capable if steer[CODES[c]]['auc_drop_remove'] >= 0.1],
       'mean_insert_neg_rise': round(float(np.mean([steer[CODES[c]]['meanprob_neg_after_insert']
                                                    - steer[CODES[c]]['meanprob_neg_base'] for c in capable])), 3),
       'top_shared_feature_pairs': pairs[:20],
       'steer': steer}
json.dump(res, open(f'{QK}/ecg_feature_causal.json', 'w'), indent=2)
json.dump(render, open(f'{QK}/ecg_feature_render.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('n_capable', 'mean_auc_drop_remove',
      'codes_remove_collapses_auc>=0.1', 'mean_insert_neg_rise')}, indent=1), flush=True)
print('TOP SHARED-FEATURE PAIRS:', json.dumps(pairs[:12], indent=1), flush=True)
print('ECG FEATURE CAUSAL DONE', flush=True)
