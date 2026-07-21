"""Depth-generalization of the composed-feature finding (Logan 2026-07-21): does layer-2 QK also
have composed (current,attended) features that beat individual token classes, and are they syntactic?
F19: deep selection is distributed, so this is falsifiable. Same method as F36 but at layer L=2
(capture through block 2; attended token = argmax layer-2 attention). Composed vs individual FVU +
decode. Compares to layer 1 (F36: composed 0.047 vs individual 0.662 at K=1024).
"""
import sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
Vsz = cfg['vocab_size']
tk = AutoTokenizer.from_pretrained('gpt2')
NSEQ = 200
TOK = build_eval_tokens(n_chunks=NSEQ, seq_len=513)[:NSEQ]
rms = lambda x: F.rms_norm(x, (D,))
L = 2
R_qk = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in ['c_q', 'c_k', 'c_q2', 'c_k2']], 0)
RANK = 128


@torch.no_grad()
def capture(idx):
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    for li in range(L + 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L:
            att = pat.abs().sum(1); att[:, torch.arange(T, device=DEV), torch.arange(T, device=DEV)] = -1
            keyt = idx.gather(1, att.argmax(-1))
            return h.reshape(-1, D), keyt.reshape(-1), idx.reshape(-1)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))


Hs, keys, curs = [], [], []
for i in range(0, NSEQ, 20):
    h, kt, ct = capture(TOK[i:i + 20, :-1].to(DEV))
    Hs.append(h); keys.append(kt); curs.append(ct)
H = torch.cat(Hs); keytok = torch.cat(keys); cur = torch.cat(curs)


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T
C = (H.double().T @ H.double()) / H.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R_qk.double().T @ R_qk.double()) @ Cs)
ENC = (Cis.float() @ U[:, -RANK:].float())
Z = H @ ENC
# between-token variance fraction (compare to layer-1's 0.82)
tm = torch.zeros(Vsz, RANK, device=DEV); ct = torch.zeros(Vsz, device=DEV)
tm.index_add_(0, cur, Z); ct.index_add_(0, cur, torch.ones(len(cur), device=DEV))
tm[ct > 0] /= ct[ct > 0][:, None]
btf = 1 - ((Z - tm[cur]) ** 2).sum().item() / ((Z - Z.mean(0)) ** 2).sum().item()

pair = cur.long() * Vsz + keytok.long()
up, inv, cnt = torch.unique(pair, return_inverse=True, return_counts=True)
pm = torch.zeros(len(up), RANK, device=DEV); pm.index_add_(0, inv, Z); pm /= cnt[:, None].float()
keep = cnt >= 8
pmk = pm[keep]; upk = up[keep]; cntk = cnt[keep].float()
print(f'layer {L}: {H.shape[0]} positions, {len(upk)} frequent pairs; between-token var frac {btf:.3f} '
      f'(layer-1 was 0.82)', flush=True)


def kmeans(X, K, iters=25):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = ((X * X).sum(1, True) - 2 * X @ Cc.T + (Cc * Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); c2 = torch.zeros(K, device=X.device)
        Cn.index_add_(0, asg, X); c2.index_add_(0, asg, torch.ones(len(X), device=X.device))
        mm = c2 > 0; Cc[mm] = Cn[mm] / c2[mm][:, None]
    return asg, Cc


tv = ((pmk - (pmk * cntk[:, None]).sum(0) / cntk.sum()) ** 2 * cntk[:, None]).sum().item()
wfvu = lambda rec: (((pmk - rec) ** 2) * cntk[:, None]).sum().item() / tv
res = {'layer': L, 'between_token_var_frac': round(btf, 3), 'n_pairs': int(len(upk)), 'composed': {}, 'individual': {}}
pck = upk // Vsz; pak = upk % Vsz
for K1 in [8, 16, 32]:
    Kp = K1 * K1
    ac2, cent = kmeans(pmk, min(Kp, len(upk))); fc = wfvu(cent[ac2])
    uc = torch.unique(pck); ua = torch.unique(pak)
    cf = torch.stack([(pmk[pck == t] * cntk[pck == t, None]).sum(0) / cntk[pck == t].sum() for t in uc])
    af = torch.stack([(pmk[pak == t] * cntk[pak == t, None]).sum(0) / cntk[pak == t].sum() for t in ua])
    acl, _ = kmeans(cf, min(K1, len(uc))); aal, _ = kmeans(af, min(K1, len(ua)))
    cmm = {int(t): int(acl[i]) for i, t in enumerate(uc)}; amm = {int(t): int(aal[i]) for i, t in enumerate(ua)}
    cell = torch.tensor([cmm[int(pck[i])] * K1 + amm[int(pak[i])] for i in range(len(upk))], device=DEV)
    cc = torch.zeros(K1 * K1, RANK, device=DEV); c3 = torch.zeros(K1 * K1, device=DEV)
    cc.index_add_(0, cell, pmk * cntk[:, None]); c3.index_add_(0, cell, cntk)
    mm = c3 > 0; cc[mm] /= c3[mm][:, None]; fi = wfvu(cc[cell])
    res['composed'][Kp] = round(fc, 4); res['individual'][Kp] = round(fi, 4)
    print(f'  K={Kp}: composed {fc:.4f} | individual {fi:.4f}', flush=True)
json.dump(res, open(f'{OUT}/bilin18_layer2_composed.json', 'w'), indent=2)
# decode
asg, _ = kmeans(F.normalize(pmk, dim=1), 64)
dec = lambda t: repr(tk.decode([int(t)]))
sizes = torch.bincount(asg, minlength=64)
print('\n--- sample layer-2 composed classes (current->attended) ---', flush=True)
for c in torch.argsort(sizes, descending=True)[:8].tolist():
    mem = torch.nonzero(asg == c).squeeze(1); mem = mem[torch.argsort(cntk[mem], descending=True)][:7]
    print(f'class {c}: ' + '  '.join(f'{dec(upk[i]//Vsz)}->{dec(upk[i]%Vsz)}' for i in mem.tolist()), flush=True)
print('bilin18 layer2 composed done', flush=True)
