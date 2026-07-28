"""ECG: three-continent validation of the DISCOVERY targets (sex, age), to match the
known-feature (BBB) three-continent result. Tests whether the discovered features
hold across Germany + US + China, and how the generalization gradient behaves at
three-cohort resolution.
"""
import glob, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'


def hdr_field(hp, key):
    for line in open(hp):
        lk = line.lower()
        if lk.startswith(f'# {key}') or lk.startswith(f'#{key}'):
            return line.split(':', 1)[1].strip()
    return None


def cohort_labels(dirname, target):
    recs = sorted(glob.glob(f'{OUT}/{dirname}/*/*.hea'))
    vals = []
    for hp in recs:
        if target == 'sex':
            v = hdr_field(hp, 'sex')
            vals.append(1.0 if (v and v.lower().startswith('f')) else (0.0 if v else -1))
        else:
            v = hdr_field(hp, 'age')
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(-1)
    return np.array(vals, dtype=np.float32)


def load_model(name):
    ck = torch.load(f'{QK}/{name}', map_location=DEV)
    cfg = ck['cfg']; W = ck['state']
    return ck, cfg, W


def make_forward(cfg, W, MU, SD):
    D, NH, HD, NL = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL']
    PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
    norm = lambda x: (x - MU) / SD

    def patch(x):
        B = x.shape[0]
        return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    A = W['embed.weight'].T @ W['blocks.1.L.weight'].T
    Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T

    @torch.no_grad()
    def feats(X):
        o = []
        for i in range(0, len(X), 2048):
            P = patch(norm(X[i:i+2048]))
            o.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
        return torch.cat(o)
    return feats


def strength(F_, y, continuous):
    out = []
    for j in range(F_.shape[1]):
        s = F_[:, j].float(); l = y.float()
        if continuous:
            r = torch.corrcoef(torch.stack([s, l]))[0, 1]
            out.append(0.0 if torch.isnan(r) else float(r.abs()))
        else:
            o = torch.argsort(torch.argsort(s)); rk = o.float()+1
            p = l.bool().sum().float(); n = (~l.bool()).sum().float()
            a = 0.5 if p == 0 or n == 0 else float((rk[l.bool()].sum()-p*(p+1)/2)/(p*n))
            out.append(abs(a-0.5))
    return np.array(out)


df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); fold = df.strat_fold.values
Xp_full = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Xg_full = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
Xc_full = torch.from_numpy(np.load(f'{OUT}/chapman_X.npy')).to(DEV)
res = {}
for target, model, continuous in (('sex', 'ecg_sex_model.pt', False), ('age', 'ecg_age_model.pt', True)):
    ck, cfg, W = load_model(model)
    feats = make_forward(cfg, W, ck['MU'].to(DEV), ck['SD'].to(DEV))
    # labels
    yp = df.sex.values[fold == 10].astype(np.float32) if target == 'sex' else df.age.values[fold == 10].astype(np.float32)
    yg = cohort_labels('georgia', target)[:len(Xg_full)]
    yc = cohort_labels('chapman_shaoxing', target)[:len(Xc_full)]
    # valid masks
    def vmask(y):
        return (y >= 0) & (y <= 95) if continuous else (y >= 0)
    mp, mg, mc = vmask(yp), vmask(yg), vmask(yc)
    Fp = strength(feats(Xp_full[torch.from_numpy(mp).to(DEV)]), torch.from_numpy(yp[mp]).to(DEV), continuous)
    Fg = strength(feats(Xg_full[torch.from_numpy(mg).to(DEV)]), torch.from_numpy(yg[mg]).to(DEV), continuous)
    Fc = strength(feats(Xc_full[torch.from_numpy(mc).to(DEV)]), torch.from_numpy(yc[mc]).to(DEV), continuous)
    res[f'{target}_corr_DE_US'] = round(float(np.corrcoef(Fp, Fg)[0, 1]), 3)
    res[f'{target}_corr_US_CN'] = round(float(np.corrcoef(Fg, Fc)[0, 1]), 3)
    res[f'{target}_corr_DE_CN'] = round(float(np.corrcoef(Fp, Fc)[0, 1]), 3)
    tri = np.minimum(np.minimum(Fp, Fg), Fc)
    top = list(np.argsort(-tri)[:6])
    res[f'{target}_3continent_units'] = [int(u) for u in top]
    res[f'{target}_3continent_strengths'] = [[round(float(Fp[j]), 3), round(float(Fg[j]), 3), round(float(Fc[j]), 3)] for j in top]
    res[f'{target}_n_US_valid'] = int(mg.sum()); res[f'{target}_n_CN_valid'] = int(mc.sum())
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_discovery3.json', 'w'), indent=2)
print('ECG DISCOVERY3 DONE', flush=True)
