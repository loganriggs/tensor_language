"""T2: SELECTION-CHANNEL CENSUS across all 18 layers x 9 heads. For each head, fit its pattern
against a small library of explicit selection predicates (the program family the induction
predicate validated), held-out R^2 per head, then GATE the nameable heads by substitution.
Predicate features per (query i, key j):
  MATCH_prev  : 1[tok_{j-1} == tok_i]      (induction-style match)
  MATCH_same  : 1[tok_j == tok_i]          (same-token attention)
  KEY_punct / KEY_func / KEY_cap : key-token class indicators
  FIRST       : 1[j == 0]
  PREV1/PREV2 : 1[j == i-1] / 1[j == i-2]  (local recency spikes)
  TEMPLATE    : the head's positional mean pattern (offset structure)
Fit per head by least squares on cooc batches; held-out R^2 on fresh cooc rows. Census table =
per-head best structure + coefficient profile. GATE: substitute coded patterns at the top-K
programmatic heads (excluding template-only heads) simultaneously; dCE with SE on held-back
FW[448:600]. Notes: template captures pure-positional heads (named 'positional'); a head is called
PROGRAMMATIC if predicates beyond template explain >=5% additional pattern variance held-out.
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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
KP = torch.zeros(V, dtype=torch.bool); KF = torch.zeros(V, dtype=torch.bool); KC = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): KP[i] = True
    if core.lower() in FUNC: KF[i] = True
    if lead and len(core) and core[0].isupper(): KC[i] = True
KP, KF, KC = KP.to(DEV), KF.to(DEV), KC.to(DEV)
T0 = 128
FEATN = ['MATCH_prev', 'MATCH_same', 'KEY_punct', 'KEY_func', 'KEY_cap', 'FIRST', 'PREV1', 'PREV2']
NF = len(FEATN)

def feats(idx):
    """(B,NF,T,T) predicate features + causal mask."""
    B, T = idx.shape
    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    prevtok = torch.roll(idx, 1, dims=1); prevtok[:, 0] = -1
    Fs = torch.zeros(B, NF, T, T, device=DEV)
    Fs[:, 0] = (prevtok.unsqueeze(1) == idx.unsqueeze(2)).float()        # tok_{j-1}==tok_i
    Fs[:, 1] = (idx.unsqueeze(1) == idx.unsqueeze(2)).float()            # tok_j==tok_i
    Fs[:, 2] = KP[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 3] = KF[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 4] = KC[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 5, :, 0] = 1.0
    eye1 = torch.diag(torch.ones(T-1, device=DEV), -1); Fs[:, 6] = eye1
    eye2 = torch.diag(torch.ones(T-2, device=DEV), -2); Fs[:, 7] = eye2
    return Fs * causal, causal


@torch.no_grad()
def patterns(idx):
    """per-layer (B,NH,T,T) patterns from the real forward."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = []
    for li in range(NL):
        b = m.transformer.h[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        out.append(pat)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))
    return out


P=64; NSEQ=48
prefN = FW[100:100+NSEQ, 1:1+P]; EVi=torch.cat([prefN,prefN],1).to(DEV)
SECi=torch.arange(P,2*P-1,device=DEV); FIRi=torch.arange(1,P-1,device=DEV)
HELDn = FW[448:600][:, :129].to(DEV)
prog=json.load(open(f'{QK}/qk_selection_census.json'))['programmatic']
c=[c for c in prog if c['layer']==3 and c['head']==8][0]
# recompute L3H8 coefficients need W; instead re-fit just L3H8 quickly is heavy -> use census top_coef for MATCH_same and MATCH_prev signs
# Directional steer: add s * (MATCH_same - baseline) to L3H8 pattern using the census-reported anti-self sign.
import numpy as _np
LT,HT=3,8
@torch.no_grad()
def fwd(idx, s):
    B,T=idx.shape; x0=F.rms_norm(m.transformer.wte(idx),(D,)); x=None; v1=None
    cos,sin=rope_tables(T,HD,DEV,x0.dtype,'bf16'); cb,sb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    Fs,causal=feats(idx)
    for li in range(NL):
        b=m.transformer.h[li]; a=b.attn
        x=(b.lambdas[0]+b.lambdas[1])*x0 if li==0 else b.lambdas[0]*x+b.lambdas[1]*x0
        h=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(h).view(B,T,NH,HD),(HD,)); return apply_rot(z,cb,sb)
        v=a.c_v(h).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        pat=((torch.einsum('bqhd,bkhd->bhqk',q,k)/HD)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD)).masked_fill(~mask,0.0)
        if li==LT:
            pat=pat.clone(); pat[:,HT]=pat[:,HT]+s*Fs[:,1]  # Fs[:,1]=MATCH_same (tok_j==tok_i); +s adds same-token attention
        yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        x=x+a.c_proj(yh.reshape(B,T,-1)); x=x+b.mlp(F.rms_norm(x,(D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30).float()
@torch.no_grad()
def adv(s):
    lg=fwd(EVi[:,:-1],s); tgt=EVi[:,1:]
    ce=F.cross_entropy(lg.reshape(-1,V),tgt.reshape(-1),reduction='none').view(NSEQ,-1)
    return float(ce[:,FIRi].mean()-ce[:,SECi].mean())
@torch.no_grad()
def natce(s):
    tot=0;n=0
    for i in range(0,len(HELDn),4):
        bb=HELDn[i:i+4]; lg=fwd(bb[:,:-1],s)
        tot+=float(F.cross_entropy(lg.reshape(-1,V),bb[:,1:].reshape(-1)))*bb[:,1:].numel(); n+=bb[:,1:].numel()
    return tot/n
res={}
for s in (-0.2,-0.1,0.0,0.1,0.2):
    res[f's={s}']={'induction_adv':round(adv(s),3),'natural_CE':round(natce(s),4)}
    print(f"L3H8 same-token steer s={s}: induction adv {res[f's={s}']['induction_adv']} | natural CE {res[f's={s}']['natural_CE']}",flush=True)
json.dump(res,open(f'{QK}/qk_l3h8_steer.json','w'),indent=2)
print("QK L3H8 STEER DONE",flush=True)
