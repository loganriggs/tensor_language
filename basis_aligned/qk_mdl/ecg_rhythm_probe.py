"""Path 2 step 4: interpret the student's RHYTHM capability + its clinical baseline. The clinical
rules: atrial fibrillation = irregularly-irregular RR (high RR variability, no P waves); sinus
bradycardia = slow rate; sinus tachycardia = fast rate. Extract R-peaks -> heart rate + RR
coefficient-of-variation, and ask: do these simple clinical metrics reproduce the student/teacher
rhythm detections? (Does the model reduce to the clinical rule, or add something?) Also which the
student's AF output correlates with: RR-irregularity (rhythm) vs P-wave region (morphology).
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

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
Yhard = (Ste > 0.5).cpu().numpy()
FS = 100  # Hz


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def student(x):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return torch.sigmoid(F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias'])


Sst = torch.cat([student(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)]).cpu().numpy()

# clinical rhythm metrics from R-peaks (lead II)
Xn = ((Xte - MU) / SD).cpu().numpy()
HR = np.zeros(len(Xn)); RRCV = np.zeros(len(Xn)); Pamp = np.zeros(len(Xn))
for i in range(len(Xn)):
    sig = Xn[i, 1]
    pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    if len(pk) >= 3:
        rr = np.diff(pk) / FS                       # seconds
        HR[i] = 60.0 / rr.mean()
        RRCV[i] = rr.std() / (rr.mean() + 1e-9)     # RR coefficient of variation (irregularity)
        # P-wave amplitude proxy: signal energy in the 120ms window before each R (P region), averaged
        pw = [np.abs(sig[p-16:p-4]).mean() for p in pk if p-16 >= 0]
        Pamp[i] = np.mean(pw) if pw else 0.0
    else:
        HR[i] = 60.0; RRCV[i] = 0.0; Pamp[i] = 0.0


def auc(score, y):
    y = y.astype(bool); p = y.sum(); n = (~y).sum()
    if p == 0 or n == 0: return 0.5
    o = np.argsort(np.argsort(score)) + 1
    return float((o[y].sum() - p*(p+1)/2) / (p*n))


# clinical-criterion baselines vs teacher labels, and how well they explain the STUDENT output
res = {'n_test': len(Xn), 'rhythm': {}}
tests = {'AF': ('RRCV', RRCV, 'irregular RR'), 'SB': ('-HR', -HR, 'slow rate'), 'ST': ('HR', HR, 'fast rate')}
for cls, (fname, feat, desc) in tests.items():
    j = TC.index(cls); y = Yhard[:, j]
    clin_auc = auc(feat, y)
    student_auc = auc(Sst[:, j], y)
    # does the clinical metric explain the student's own output? (rank corr)
    sc = float(np.corrcoef(feat, Sst[:, j])[0, 1])
    res['rhythm'][cls] = {'clinical_metric': fname, 'desc': desc,
                          'clinical_auc_vs_teacher': round(clin_auc, 3),
                          'student_auc_vs_teacher': round(student_auc, 3),
                          'corr_metric_vs_student_output': round(sc, 3)}
    print(f'  {cls} ({desc}): clinical-metric AUC {clin_auc:.3f} | student AUC {student_auc:.3f} | '
          f'corr(metric,student) {sc:+.3f}', flush=True)
# AF specifically: RR-irregularity (rhythm) vs P-wave absence (morphology) — which explains student AF better
jAF = TC.index('AF')
res['AF_mechanism'] = {'corr_RRCV_vs_studentAF': round(float(np.corrcoef(RRCV, Sst[:, jAF])[0, 1]), 3),
                       'corr_Pabsence_vs_studentAF': round(float(np.corrcoef(-Pamp, Sst[:, jAF])[0, 1]), 3),
                       'RRCV_auc_vs_teacherAF': round(auc(RRCV, Yhard[:, jAF]), 3),
                       'Pabsence_auc_vs_teacherAF': round(auc(-Pamp, Yhard[:, jAF]), 3)}
print('AF mechanism -> RR-irregularity vs P-absence:', json.dumps(res['AF_mechanism']), flush=True)
json.dump(res, open(f'{QK}/ecg_rhythm_probe.json', 'w'), indent=2)
print('ECG RHYTHM PROBE DONE', flush=True)
