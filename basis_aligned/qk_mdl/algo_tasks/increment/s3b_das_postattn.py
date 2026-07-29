"""Step 3b: DAS-lite at the POST-attention residual of layer 8 (site 'pre_mlp', li=8),
final position — the point where the top component ('h',8,7) has just written.

Motivated by s3's negative result: the increment payload travels via the v1 cache
(layer-0 values read by layer-8 heads), so it is NOT present in the pre-attention
residual at either the digit positions or the final position. It first enters the
residual stream when c_proj(L8 attn) writes at the final position.
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, forward, OUT

torch.manual_seed(0)
m, cfg = get_model()
D = cfg['n_embd']
LAYER = 8
SITE = 'pre_mlp'
POS = [7]
S = torch.load(f'{OUT}/stimuli.pt')
clean, corr = S['clean'].cuda(), S['corr'].cuda()
ca, xa = S['clean_ans'].cuda(), S['corr_ans'].cuda()

clean_resid = []
with torch.no_grad():
    for i in range(0, 40, 8):
        cch = {}
        forward(m, clean[i:i+8], cache=cch)
        clean_resid.append(cch[('resid_m', LAYER)])
clean_resid = torch.cat(clean_resid, 0)


def hook_factory(Bmat, src):
    def hook(site, li, x):
        if site == SITE and li == LAYER:
            x = x.clone()
            delta = src[:, POS] - x[:, POS]
            x[:, POS] = x[:, POS] + delta @ Bmat @ Bmat.T
        return x
    return hook


def evaluate(Bmat, sl):
    flips, Mp, Mc, Mx = [], [], [], []
    with torch.no_grad():
        for i in range(sl.start, sl.stop, 8):
            j = min(i + 8, sl.stop)
            lg = forward(m, corr[i:j], resid_hook=hook_factory(Bmat, clean_resid[i:j]))[:, -1].float()
            lgc = forward(m, clean[i:j])[:, -1].float()
            lgx = forward(m, corr[i:j])[:, -1].float()
            n = torch.arange(j - i, device='cuda')
            flips.append((lg.argmax(-1) == ca[i:j]).float())
            Mp.append(lg[n, ca[i:j]] - lg[n, xa[i:j]])
            Mc.append(lgc[n, ca[i:j]] - lgc[n, xa[i:j]])
            Mx.append(lgx[n, ca[i:j]] - lgx[n, xa[i:j]])
    flips = torch.cat(flips); Mp = torch.cat(Mp); Mc = torch.cat(Mc); Mx = torch.cat(Mx)
    return flips.mean().item(), ((Mp - Mx) / (Mc - Mx)).mean().item()


def train_subspace(r, steps=120, lr=5e-3, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = torch.randn(D, r, generator=g).cuda() * 0.02
    P.requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    order = list(range(0, 30, 8))
    for step in range(steps):
        i = order[step % len(order)]
        j = min(i + 8, 30)
        Bmat, _ = torch.linalg.qr(P)
        lg = forward(m, corr[i:j], resid_hook=hook_factory(Bmat, clean_resid[i:j]))[:, -1]
        loss = F.cross_entropy(lg.float(), ca[i:j])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Bmat, _ = torch.linalg.qr(P)
    return Bmat.detach()


res = {}
fh, rh = evaluate(torch.eye(D).cuda(), slice(30, 40))
res['fullresid'] = {'held_flip': round(fh, 3), 'held_rf': round(rh, 4)}
print(f'[postattn-final] FULL residual patch: held flip {fh:.3f}  rf {rh:.4f}', flush=True)
for r in [1, 4, 16]:
    Bmat = train_subspace(r)
    ftr, rtr = evaluate(Bmat, slice(0, 30))
    fhe, rhe = evaluate(Bmat, slice(30, 40))
    fls, rfs = [], []
    for sd in range(5):
        g = torch.Generator(device='cpu').manual_seed(100 + sd)
        R, _ = torch.linalg.qr(torch.randn(D, r, generator=g).cuda())
        f0, r0 = evaluate(R, slice(30, 40))
        fls.append(f0); rfs.append(r0)
    res[f'r{r}'] = {'train_flip': round(ftr, 3), 'train_rf': round(rtr, 4),
                    'held_flip': round(fhe, 3), 'held_rf': round(rhe, 4),
                    'random_ctrl_held_flip': round(sum(fls)/5, 3),
                    'random_ctrl_held_rf': round(sum(rfs)/5, 4)}
    print(f'[postattn-final] r={r}: train flip {ftr:.3f}/rf {rtr:.3f} | HELD flip {fhe:.3f}/rf {rhe:.3f} '
          f'| random ctrl flip {sum(fls)/5:.3f}/rf {sum(rfs)/5:.4f}', flush=True)

json.dump(res, open(f'{OUT}/s3b_das_postattn.json', 'w'), indent=2)
print('saved s3b_das_postattn.json')
