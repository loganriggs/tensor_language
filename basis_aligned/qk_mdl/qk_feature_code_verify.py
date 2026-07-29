"""VERIFY FEATURE MEANING BY CODE (Logan's loop): hypothesis 'feature r detects class C' is written
as literal code f_r = alpha*1[token in C] + beta with a HAND-WRITTEN grammar list (not derived from
the feature), translated back into the model (inside the MLP0 program), then verified: (a) the tasks
the feature should serve still work; (b) other tasks don't break; (c) the 2 scalars/feature can be
finetuned (non-function-affecting knob). Variants: learned program | zero top-8 features | coded
top-8 | coded finetuned. feat48 (topical nouns) has NO closed-class code -> negative control (its
'code' is a generic content-word list; if that works, the method is too permissive).
"""
import json, sys
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
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
A0, U0 = torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)['table_R256']

# hand-written grammar classes (hypotheses from the feature readouts; grammar-derived lists)
CLASSES = {
    174: {'name': 'conjunctions', 'words': {'and','or','but','because','while','as','though','although','which','whereas','nor','so','yet','since','if','when','unless','whether','&'}},
    28:  {'name': 'determiners/particles', 'words': {'a','an','the','to','his','her','my','your','their','its','our','be','not','this','that','these','those','some','any','no','each','every'}},
    75:  {'name': 'be-verbs/pronouns', 'words': {'are','is','was','were','am','been','being','her','my','your','the','you','we','they','he','she','it','a'}},
    184: {'name': 'linking words', 'words': {'as','and','or','be','are','is','but','was','were','being'}},
    96:  {'name': 'prepositions', 'words': {'to','in','on','for','with','as','at','by','from','of','into','over','under','about','through','between','during','against','among'}},
    156: {'name': 'det/prep core', 'words': {'to','the','of','a','his','an','their','for','my','her','its','our','in'}},
    205: {'name': 'pronouns', 'words': {'you','your','they','we','my','he','i','she','it','us','them','our','his','her','me','him'}},
    48:  {'name': 'topical nouns (NEG CONTROL)', 'words': {'system','government','world','people','information','research','technology','business','water','energy','health','market','history','science'}},
}
CODED = sorted(CLASSES.keys())
# membership vectors over vocab
MEMB = {}
for r_, spec in CLASSES.items():
    v = torch.zeros(V, dtype=torch.bool)
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        if s.replace('Ġ', '').lower() in spec['words']: v[i] = True
    MEMB[r_] = v.to(DEV)
    print(f"feat{r_} ({spec['name']}): {int(v.sum())} vocab tokens", flush=True)

import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which','you','have','he','they','has'}
def masks():
    ms = {k: torch.zeros(V, dtype=torch.bool) for k in ['subword','punct','capital','digit','funcword']}
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
        if not lead and len(core) and core[0].isalpha() and core[0].islower(): ms['subword'][i] = True
        if len(core) and all(c in _P for c in core): ms['punct'][i] = True
        if lead and len(core) and core[0].isupper(): ms['capital'][i] = True
        if len(core) and all(c.isdigit() for c in core): ms['digit'][i] = True
        if core.lower() in FUNC: ms['funcword'][i] = True
    return {k: v.to(DEV) for k, v in ms.items()}
MASKS = masks()


@torch.no_grad()
def block0(idx):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
    return hin, blk.mlp(hin)

# token table + calibrate alpha,beta per coded feature (lstsq of learned activation on membership)
Hs, Ys, Ts = [], [], []
for i in range(0, 240, 8):
    h, y = block0(COOC[i:i+8].to(DEV)[:, :128]); Hs.append(h.reshape(-1, D)); Ys.append(y.reshape(-1, D)); Ts.append(COOC[i:i+8, :128].reshape(-1).to(DEV))
H = torch.cat(Hs); Y = torch.cat(Ys); T = torch.cat(Ts)
ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
ts.index_add_(0, T, Y); tc.index_add_(0, T, torch.ones_like(T, dtype=torch.float32))
lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0); TT0 = lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*Y.mean(0)
Fl = (H @ A0.T)**2
AB = {}
for r_ in CODED:
    mvec = MEMB[r_][T].float()
    X = torch.stack([mvec, torch.ones_like(mvec)], 1)
    sol = torch.linalg.lstsq(X, Fl[:, r_].unsqueeze(1)).solution.squeeze(1)
    fitr2 = 1 - (X @ sol - Fl[:, r_]).pow(2).sum()/(Fl[:, r_] - Fl[:, r_].mean()).pow(2).sum()
    AB[r_] = sol
    print(f"feat{r_}: alpha {sol[0]:.1f} beta {sol[1]:.1f}  code-vs-learned R2 {fitr2:.3f}", flush=True)

CIDX = torch.tensor(CODED, device=DEV)
def prog_out(hin_flat, tok_flat, mode, ab):
    Fq = (hin_flat @ A0.T)**2
    if mode == 'zero8':
        Fq[:, CIDX] = 0.0
    elif mode in ('coded8', 'tuned8'):
        for j, r_ in enumerate(CODED):
            Fq[:, r_] = ab[j, 0]*MEMB[r_][tok_flat].float() + ab[j, 1]
    return TT0[tok_flat] + Fq @ U0

def forward(idx, mode, ab=None):
    B, T2 = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T2, -1)); hin = F.rms_norm(x, (D,))
        if li == 0 and mode != 'model':
            mo = prog_out(hin.reshape(-1, D), idx.reshape(-1), mode, ab).view(B, T2, D).to(x.dtype)
            x = x + mo
        else:
            x = x + blk.mlp(hin)
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)

@torch.no_grad()
def metrics(mode, ab=None):
    idxN = FINEWEB[:64, :128].to(DEV)
    lg = forward(idxN[:, :-1], mode, ab).float(); tgt = idxN[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(64, -1)
    out = {'natural_CE': round(ce.mean().item(), 4)}
    for t in MASKS: out[t] = round(ce[MASKS[t][tgt]].mean().item(), 4)
    lg = forward(EVN[:, :-1], mode, ab).float()
    cei = F.cross_entropy(lg.reshape(-1, V), EVN[:, 1:].reshape(-1), reduction='none').view(NSEQ, -1)
    out['induction_adv'] = round(cei[:, FIR].mean().item() - cei[:, SEC].mean().item(), 3)
    return out

ab0 = torch.stack([AB[r_] for r_ in CODED])
res = {}
for mode, ab in [('model', None), ('learned', None), ('zero8', None), ('coded8', ab0)]:
    res[mode] = metrics(mode, ab); print(mode, res[mode], flush=True)

# non-function-affecting finetune: tune 16 scalars on natural CE
abp = torch.nn.Parameter(ab0.clone())
opt = torch.optim.Adam([abp], lr=1e-2)
for step in range(120):
    i = np.random.randint(0, 5000); b = COOC[i:i+2].to(DEV)[:, :128]
    lg = forward(b[:, :-1], 'tuned8', abp).float()
    loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
res['coded8_tuned'] = metrics('tuned8', abp.detach()); print('coded8_tuned', res['coded8_tuned'], flush=True)
json.dump(res, open(f'{QK}/qk_feature_code_verify.json', 'w'), indent=2)
print("QK FEATURE CODE VERIFY DONE", flush=True)
