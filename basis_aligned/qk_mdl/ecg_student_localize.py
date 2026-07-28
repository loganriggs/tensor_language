"""Path 2 step 3: interpret the distilled student. Novel question — HOW does a time-patched
foldable model compute RHYTHM (AF/SB/ST, which need integrating across many beats) vs MORPHOLOGY
(LBBB/RBBB, single-beat shape)? Hypothesis: rhythm needs ATTENTION (mixes time-patches) while
morphology needs the MLPs (per-patch). Ablate each attention layer and each MLP layer, measure
per-class AUC drop (student output vs teacher hard label). Also whole-attention vs whole-MLP.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
ck = torch.load(f'{QK}/ecg_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; TC = ck['classes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Ste = torch.from_numpy(np.load(f'{QK}/teacher_soft_test.npy')).to(DEV)
Yhard = (Ste > 0.5).float()


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, kill_attn=None, kill_mlp=None):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        if kill_attn != li:
            def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
            q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
            h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        if kill_mlp != li:
            h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def all_auc(**kw):
    s = torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()
    R = torch.argsort(torch.argsort(s, 0), 0).float() + 1
    out = np.zeros(NCLS)
    for c in range(NCLS):
        lab = Yhard[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
        out[c] = 0.5 if p == 0 or n == 0 else float((R[lab, c].sum()-p*(p+1)/2)/(p*n))
    return out


base = all_auc()
# per-layer ablations (kill_attn/kill_mlp expect a real layer index; None disables)
attn_drop = np.stack([base - all_auc(kill_attn=L) for L in range(NL)])   # (NL,6)
mlp_drop = np.stack([base - all_auc(kill_mlp=L) for L in range(NL)])     # (NL,6)

RHYTHM = ['SB', 'AF', 'ST']; MORPH = ['1dAVb', 'RBBB', 'LBBB']
res = {'base_auc': {TC[c]: round(float(base[c]), 3) for c in range(NCLS)},
       'per_class': {}}
for c in range(NCLS):
    res['per_class'][TC[c]] = {'attn_layer_drops': [round(float(attn_drop[L, c]), 3) for L in range(NL)],
                               'mlp_layer_drops': [round(float(mlp_drop[L, c]), 3) for L in range(NL)],
                               'total_attn': round(float(attn_drop[:, c].sum()), 3),
                               'total_mlp': round(float(mlp_drop[:, c].sum()), 3)}
    e = res['per_class'][TC[c]]
    print(f"  {TC[c]:6s}: attn {e['attn_layer_drops']} (tot {e['total_attn']}) | mlp {e['mlp_layer_drops']} (tot {e['total_mlp']})", flush=True)
# rhythm vs morphology attention-reliance
ratt_r = float(np.mean([res['per_class'][c]['total_attn'] for c in RHYTHM]))
rmlp_r = float(np.mean([res['per_class'][c]['total_mlp'] for c in RHYTHM]))
ratt_m = float(np.mean([res['per_class'][c]['total_attn'] for c in MORPH]))
rmlp_m = float(np.mean([res['per_class'][c]['total_mlp'] for c in MORPH]))
res['rhythm_mean_attn_drop'] = round(ratt_r, 3); res['rhythm_mean_mlp_drop'] = round(rmlp_r, 3)
res['morph_mean_attn_drop'] = round(ratt_m, 3); res['morph_mean_mlp_drop'] = round(rmlp_m, 3)
res['hypothesis_rhythm_needs_attention'] = bool(ratt_r > rmlp_r and ratt_r > ratt_m)
json.dump(res, open(f'{QK}/ecg_student_localize.json', 'w'), indent=2)
print(f'RHYTHM: attn-drop {ratt_r:.3f} vs mlp-drop {rmlp_r:.3f} | MORPH: attn-drop {ratt_m:.3f} vs mlp-drop {rmlp_m:.3f}', flush=True)
print(f"rhythm-needs-attention (attn>mlp for rhythm AND attn>morph-attn): {res['hypothesis_rhythm_needs_attention']}", flush=True)
print('ECG STUDENT LOCALIZE DONE', flush=True)
