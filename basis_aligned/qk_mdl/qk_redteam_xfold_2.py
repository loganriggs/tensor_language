"""Follow-up to qk_redteam_xfold.py ATTACK 1: complete the block-0 routing statement.
The main script substituted block 0's decayed mlp write out of the FEED-FORWARD inputs of
blocks 1/2/3 (gauge free). Block 0 could also be consumed via the ATTENTION inputs of those
blocks. Here: mean-replace block 0's decayed mlp-write contribution to block L's attention
input (hcur recomputed from the modified stream; the attention block's q/k/v all see it),
everything else untouched, L in {1, 2, 3}. Held FW[448:600,:128], paired SEs, batch 4, <4GB.
Appends 'attack1_attention_route' to qk_redteam_xfold.json."""
import json, os, sys, time, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_redteam_xfold.json'

def gpu_guard(min_free=4500, tries=90, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 4
S_, T_ = HELD.shape
LI3 = 3

@torch.no_grad()
def fwd(idx, mode=None, attn_layer=None, M0MEANS=None, stats=None):
    """VERBATIM skeleton from qk_redteam_xfold.fwd3; 'attnabl' mean-replaces the decayed M0
    stream in block attn_layer's attention input (hcur); 'collect' gathers M0 stream means at
    the ATTENTION-input point of layers 1..3 (post-decay, pre-aout)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track: Ml = []
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn
        if track and li <= LI3:
            Ml = [blk.lambdas[0]*mm for mm in Ml]
        hcur = F.rms_norm(x, (D,))
        if track and mode == 'collect' and li in (1, 2, 3):
            stats['m0sum_attn'][li] += Ml[0].sum(0)
        if track and mode == 'attnabl' and li == attn_layer:
            hcur = F.rms_norm(x - Ml[0] + M0MEANS[li].unsqueeze(0), (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
        if track and li < LI3: Ml.append(mo)
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

print("collect M0 attention-input means + base ...", flush=True)
st = {'m0sum_attn': {1: torch.zeros(T_, D, device=DEV), 2: torch.zeros(T_, D, device=DEV),
                     3: torch.zeros(T_, D, device=DEV)}}
ces = []
for i in range(0, S_, B0):
    ces.append(fwd(HELD[i:i+B0], mode='collect', stats=st).cpu())
base = torch.cat(ces, 0)
M0MEANS = {li: st['m0sum_attn'][li]/S_ for li in (1, 2, 3)}
print(f"base CE {float(base.mean()):.4f}", flush=True)

res = json.load(open(OUT))
out = {}
for L in (1, 2, 3):
    cel = []
    for i in range(0, S_, B0):
        cel.append(fwd(HELD[i:i+B0], mode='attnabl', attn_layer=L, M0MEANS=M0MEANS).cpu())
    d = (torch.cat(cel, 0) - base).flatten().double()
    mn, se = float(d.mean()), float(d.std()/np.sqrt(d.numel()))
    out[f'L{L}_attention_input_meanreplace'] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
    print(f"  attention input L{L}: dCE {mn:+.5f} +- {se:.5f}", flush=True)
res['attack1_attention_route'] = out
json.dump(res, open(OUT, 'w'), indent=1)
print("QK REDTEAM XFOLD 2 DONE", flush=True)
