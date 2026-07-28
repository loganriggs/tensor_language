"""ECG injection specificity control (negative control for §35's causal insert). A positive
causal result needs: (a) SPECIFICITY - the template raises its OWN code more than others;
(b) MORPHOLOGY-SPECIFICITY - a SCRAMBLED template (same per-lead amplitude, shape destroyed)
does NOT reproduce the effect. If a time-scrambled template moves the code just as much, the
effect was "added energy", not the morphology. Test on the strong-insert codes at a fixed
mid-range dose.
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
inj = json.load(open(f'{QK}/ecg_input_inject.json'))

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xte_n = (Xte - MU) / SD
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV); NTE = Yte.shape[0]


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward_xn(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
        h = h + (inner @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def probs_of(xn):
    return torch.cat([torch.sigmoid(forward_xn(xn[i:i+2048])) for i in range(0, len(xn), 2048)]).float()


@torch.no_grad()
def hn0_of(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    return F.rms_norm(h, (D,))


with torch.no_grad():
    HN = torch.cat([hn0_of(Xte_n[i:i+2048]) for i in range(0, len(Xte_n), 2048)])
    act = (HN @ Ahat).pow(2)


def template(r, k=300):
    flat = act[:, :, r].reshape(-1); kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        T += Xte_n[e, :, p*PT:(p+1)*PT]
    return T / kk


def scramble(T):
    # destroy waveform SHAPE, keep per-lead amplitude distribution: permute time within each lead
    out = T.clone()
    for L in range(NLEAD):
        out[L] = T[L][torch.randperm(PT, device=DEV)]
    return out


base_p = probs_of(Xte_n)
base = np.array([1.0]*NCLS)
def auc_col(s, c):
    lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))
capable = [c for c in range(NCLS) if auc_col(base_p, c) >= 0.75 and int(Yte[:, c].sum()) >= 10]
# strong-insert codes from §35
strong = [c['code'] for c in [{'code': k} for k in inj['codes_insert_rise>=0.1']]]
ALPHA = 2.0

out = {}
for code in strong:
    c = CODES.index(code)
    r = int(np.argmax(fc[:, c]))
    T = template(r); Ts = scramble(T)
    negmask = ~Yte[:, c].bool()
    def insert(TT):
        xn = (Xte_n.reshape(NTE, NLEAD, NP, PT) + ALPHA * TT[None, :, None, :]).reshape(NTE, NLEAD, NP*PT)
        return probs_of(xn)
    p_real, p_scr = insert(T), insert(Ts)
    d_real = {CODES[cc]: float(p_real[negmask, cc].mean() - base_p[negmask, cc].mean()) for cc in capable}
    tgt_real = d_real[code]
    tgt_scr = float(p_scr[negmask, c].mean() - base_p[negmask, c].mean())
    # specificity: rank of target among all codes' rises
    rank = 1 + sum(1 for cc in capable if d_real[CODES[cc]] > tgt_real)
    top3 = sorted(d_real.items(), key=lambda kv: -kv[1])[:3]
    out[code] = {'top_feature': r, 'alpha': ALPHA,
                 'target_rise_real': round(tgt_real, 3), 'target_rise_scrambled': round(tgt_scr, 3),
                 'morphology_specific': bool(tgt_real > 2 * max(tgt_scr, 0.001)),
                 'specificity_rank': rank, 'n_codes': len(capable),
                 'top3_raised': [(cc, round(v, 3)) for cc, v in top3]}
    print(f'  {code}: real +{tgt_real:.3f} vs scrambled +{tgt_scr:.3f} '
          f'(morph-specific {out[code]["morphology_specific"]}); rank {rank}/{len(capable)}; top3 {out[code]["top3_raised"]}', flush=True)

nspec = sum(1 for v in out.values() if v['morphology_specific'])
ntop = sum(1 for v in out.values() if v['specificity_rank'] <= 3)
res = {'alpha': ALPHA, 'n_tested': len(out),
       'n_morphology_specific': nspec, 'n_target_in_top3': ntop,
       'per_code': out}
json.dump(res, open(f'{QK}/ecg_inject_control.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('n_tested', 'n_morphology_specific', 'n_target_in_top3')}, indent=1), flush=True)
print('ECG INJECT CONTROL DONE', flush=True)
