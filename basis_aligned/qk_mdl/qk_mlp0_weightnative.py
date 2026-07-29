"""Logan's question: why data-estimate the program when the fold gives the exact weight tensor?
Measure it. Four arms for MLP0 at R=256, differing ONLY in what information source picks the
features/tables (substitution dCE vs the 3.63 floor):
  W1 WEIGHTS-ONLY: features = top-R eigvecs of G = sum_p ||Down_p|| (l_p l_p^T + r_p r_p^T)
     (isotropic input metric); output map CLOSED FORM u_r = Down((L a_r)*(R a_r)) (exact Frobenius
     projection of the folded tensor); table = weight-native: hin_t ~ rms(wte_t) by rms scale-
     invariance of the embedding path (ignores attn0 contribution -- stated), table_t =
     Down((L e_t)*(R e_t)) + bias.
  W2 WEIGHTS+UNIGRAM: same, but feature directions chosen under the unigram embedding second-moment
     metric M = E_t p_t e_t e_t^T (token frequencies = the only corpus statistic).
  W3 WEIGHT DIRECTIONS + DATA CALIBRATION: W2's directions, but output map U and tables fit on
     activation data (the standard recipe) -- isolates 'where features come from' vs 'how weighted'.
  DATA (reference): the committed data-fit program (+0.0748, 97.9%).
If W1/W2 approach DATA, weight-folding suffices and the data was a convenience. The measured gaps
say exactly which information the weights cannot supply (input distribution through the rms +
attention composition) vs can.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
FLOOR = 3.62671; R = 256
mlp0 = m.transformer.h[0].mlp
Lw = mlp0.Left.weight.detach().float()    # (4608, D)
Rw = mlp0.Right.weight.detach().float()
Dw = mlp0.Down.weight.detach().float()    # (D, 4608)
bias = mlp0.Down_bias.detach().float()
wte = m.transformer.wte.weight.detach().float().to(DEV)
E = F.rms_norm(wte, (D,))                  # (V, D) rms-normed embeddings

# unigram frequencies (token counts -- the only corpus statistic W2 uses)
cnt = torch.zeros(V, device=DEV); cnt.index_add_(0, COOC[:400, :128].reshape(-1).to(DEV), torch.ones(400*128, device=DEV))
p = cnt / cnt.sum()

def features_from_gram(G):
    ev, evec = torch.linalg.eigh(G)
    A = evec[:, ev.argsort(descending=True)[:R]].T.contiguous()      # (R, D) orthonormal-ish rows
    A = A / A.norm(dim=1, keepdim=True)
    return A

def closed_form_U(A):
    """Frobenius projection of the folded tensor onto span{a_r a_r^T}: solve Gram C = targets."""
    t = Dw @ ((Lw @ A.T) * (Rw @ A.T))                               # (D, R): T(a_r,a_r,:)
    Gram = (A @ A.T) ** 2                                            # (R, R)
    C = torch.linalg.solve(Gram + 1e-6*torch.eye(R, device=DEV), t.T)  # (R, D)
    return C

wp = Dw.norm(dim=0)                                                  # (4608,) output importance per neuron
G_iso = torch.einsum('p,pi,pj->ij', wp, Lw, Lw) + torch.einsum('p,pi,pj->ij', wp, Rw, Rw)
A1 = features_from_gram(G_iso); U1 = closed_form_U(A1)
# W2: unigram embedding metric
M = torch.einsum('v,vi,vj->ij', p, E, E)
Ms = torch.linalg.cholesky(M + 1e-4*torch.eye(D, device=DEV))
G_uni = Ms.T @ G_iso @ Ms
B = features_from_gram(G_uni); A2 = (B @ Ms.T); A2 = A2 / A2.norm(dim=1, keepdim=True)
U2 = closed_form_U(A2)
# weight-native table: e_t as the (embedding-only) input approximation
TW = (Dw @ ((Lw @ E.T) * (Rw @ E.T))).T + bias                       # (V, D)
print("weight-native objects built", flush=True)

# data pass for W3 calibration + reference table (standard recipe, cooc 0-240)
@torch.no_grad()
def block0_pairs(idx):
    B_, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B_, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B_, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh4.reshape(B_, T, -1)); hin = F.rms_norm(x, (D,))
    return hin.reshape(-1, D), blk.mlp(hin).reshape(-1, D), idx.reshape(-1)
Hs, Ys, Ts = [], [], []
for i in range(0, 240, 8):
    h, y, t = block0_pairs(COOC[i:i+8].to(DEV)[:, :128]); Hs.append(h); Ys.append(y); Ts.append(t)
H = torch.cat(Hs); Y = torch.cat(Ys); T_ = torch.cat(Ts)
ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
ts.index_add_(0, T_, Y); tc.index_add_(0, T_, torch.ones_like(T_, dtype=torch.float32))
lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0); TT = lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*Y.mean(0)
# W3: A2 directions, U + implicit tables by data least squares on residual after data table
resid = Y - TT[T_]
F2 = (H @ A2.T)**2
U3 = torch.linalg.lstsq(F2, resid).solution                          # (R, D)
print("W3 calibrated", flush=True)

ARMS = {'W1_weights_only': (A1, U1, 'TW'), 'W2_weights_unigram': (A2, U2, 'TW'), 'W3_wdirs_datacal': (A2, U3, 'TT')}

@torch.no_grad()
def audit(arm):
    A_, U_, tab = ARMS[arm]
    TAB = TW if tab == 'TW' else TT
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B_, T2 = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = blk.lambdas[0]*x + blk.lambdas[1]*x0; hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B_, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B_, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B_, T2, -1)); hin = F.rms_norm(x, (D,))
            if li == 0:
                flat = hin.reshape(-1, D)
                mo = TAB[idx.reshape(-1)]
                if arm == 'W3_wdirs_datacal':
                    mo = mo + ((flat @ A_.T)**2) @ U_
                else:
                    # weight-native: table already contains the e_t-quadratic; add feature CORRECTION
                    # relative to it: features act on the actual input
                    mo = mo + ((flat @ A_.T)**2) @ U_ - (((F.rms_norm(wte, (D,))[idx.reshape(-1)]) @ A_.T)**2) @ U_
                x = x + mo.view(B_, T2, D).to(x.dtype)
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {'reference_data_program': {'dCE': 0.07480, 'frac': 0.979}}
for arm in ARMS:
    d = audit(arm) - SUBBASE
    res[arm] = {'dCE': round(d, 5), 'frac': round(1 - d/FLOOR, 3)}
    print(f"{arm}: dCE +{d:.5f} -> {1-d/FLOOR:.1%} of floor", flush=True)
json.dump(res, open(f'{QK}/qk_mlp0_weightnative.json', 'w'), indent=2)
print("QK MLP0 WEIGHTNATIVE DONE", flush=True)
