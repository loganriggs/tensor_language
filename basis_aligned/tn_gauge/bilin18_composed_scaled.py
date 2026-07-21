"""Harden F35 with more data (Logan 2026-07-21): the composed pair-feature decode used only 264
frequent pairs. Capture through layer 1 only (cheap), use many more sequences, get more frequent
pairs, re-decode the syntactic-dependency classes, and re-confirm composed<individual FVU on the
larger set. Data-validated (real attention argmax).
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
L = 1
R_qk = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in ['c_q', 'c_k', 'c_q2', 'c_k2']], 0)
RANK = 128


@torch.no_grad()
def capture_through_L1(idx):
    """run blocks 0..1, return layer-1 QK input h (Ntok,D), attention argmax key token, current token."""
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    for li in range(2):
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
            att = pat.abs().sum(1)
            att[:, torch.arange(T, device=DEV), torch.arange(T, device=DEV)] = -1
            keyt = idx.gather(1, att.argmax(-1))
            return h.reshape(-1, D), keyt.reshape(-1), idx.reshape(-1)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))


Hs, keys, curs = [], [], []
for i in range(0, NSEQ, 20):
    b = TOK[i:i + 20, :-1].to(DEV)
    h, kt, ct = capture_through_L1(b)
    Hs.append(h); keys.append(kt); curs.append(ct)
H = torch.cat(Hs); keytok = torch.cat(keys); cur = torch.cat(curs)
print(f'{H.shape[0]} positions', flush=True)


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T
C = (H.double().T @ H.double()) / H.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R_qk.double().T @ R_qk.double()) @ Cs)
ENC = (Cis.float() @ U[:, -RANK:].float())
Z = H @ ENC
pair = cur.long() * Vsz + keytok.long()
up, inv, cnt = torch.unique(pair, return_inverse=True, return_counts=True)
pm = torch.zeros(len(up), RANK, device=DEV); pm.index_add_(0, inv, Z); pm /= cnt[:, None].float()
keep = cnt >= 8
pmk = pm[keep]; upk = up[keep]; cntk = cnt[keep].float()
print(f'{len(upk)} frequent pairs (>=8 occ)', flush=True)


def kmeans(X, K, iters=25):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = ((X * X).sum(1, True) - 2 * X @ Cc.T + (Cc * Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, asg, X); ct.index_add_(0, asg, torch.ones(len(X), device=X.device))
        mm = ct > 0; Cc[mm] = Cn[mm] / ct[mm][:, None]
    return asg, Cc


# re-confirm composed vs individual FVU on the larger set
tv = ((pmk - (pmk * cntk[:, None]).sum(0) / cntk.sum()) ** 2 * cntk[:, None]).sum().item()
def wfvu(rec):
    return (((pmk - rec) ** 2) * cntk[:, None]).sum().item() / tv
res = {'n_pairs': int(len(upk)), 'composed': {}, 'individual': {}}
pck = upk // Vsz; pak = upk % Vsz
for K1 in [8, 16, 32]:
    Kp = K1 * K1
    asg_c, cent = kmeans(pmk, min(Kp, len(upk))); fc = wfvu(cent[asg_c])
    uc = torch.unique(pck); ua = torch.unique(pak)
    cf = torch.stack([(pmk[pck == t] * cntk[pck == t, None]).sum(0) / cntk[pck == t].sum() for t in uc])
    af = torch.stack([(pmk[pak == t] * cntk[pak == t, None]).sum(0) / cntk[pak == t].sum() for t in ua])
    ac, _ = kmeans(cf, min(K1, len(uc))); aa, _ = kmeans(af, min(K1, len(ua)))
    cm = {int(t): int(ac[i]) for i, t in enumerate(uc)}; am = {int(t): int(aa[i]) for i, t in enumerate(ua)}
    cell = torch.tensor([cm[int(pck[i])] * K1 + am[int(pak[i])] for i in range(len(upk))], device=DEV)
    cc = torch.zeros(K1 * K1, RANK, device=DEV); ct = torch.zeros(K1 * K1, device=DEV)
    cc.index_add_(0, cell, pmk * cntk[:, None]); ct.index_add_(0, cell, cntk)
    mm = ct > 0; cc[mm] /= ct[mm][:, None]; fi = wfvu(cc[cell])
    res['composed'][Kp] = round(fc, 4); res['individual'][Kp] = round(fi, 4)
    print(f'  K={Kp}: composed FVU {fc:.4f} | individual {fi:.4f}', flush=True)
json.dump(res, open(f'{OUT}/bilin18_composed_scaled.json', 'w'), indent=2)

# decode 64 classes
K = 64
asg, _ = kmeans(F.normalize(pmk, dim=1), K)
dec = lambda t: repr(tk.decode([int(t)]))
lines = [f'# Composed pair-features (scaled: {len(upk)} frequent pairs, {K} classes)\n',
         'Layer-1 (current→attended) equivalence classes by joint QK code — syntactic dependencies.\n']
sizes = torch.bincount(asg, minlength=K)
for c in torch.argsort(sizes, descending=True)[:30].tolist():
    mem = torch.nonzero(asg == c).squeeze(1)
    mem = mem[torch.argsort(cntk[mem], descending=True)][:12]
    lines.append(f'- **class {c}** ({int(sizes[c])}): ' + '  '.join(f'{dec(upk[i]//Vsz)}→{dec(upk[i]%Vsz)}' for i in mem.tolist()))
open(f'{OUT}/composed_pair_features_scaled.md', 'w').write('\n'.join(lines) + '\n')
print(f'\nwrote composed_pair_features_scaled.md; composed<=individual all K: '
      f'{all(res["composed"][k] <= res["individual"][k] for k in res["composed"])}', flush=True)
for c in torch.argsort(sizes, descending=True)[:6].tolist():
    mem = torch.nonzero(asg == c).squeeze(1); mem = mem[torch.argsort(cntk[mem], descending=True)][:7]
    print(f'class {c}: ' + '  '.join(f'{dec(upk[i]//Vsz)}->{dec(upk[i]%Vsz)}' for i in mem.tolist()), flush=True)
print('bilin18 composed scaled done', flush=True)
