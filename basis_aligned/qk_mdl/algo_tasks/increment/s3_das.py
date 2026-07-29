"""Step 3: DAS-lite at layer 8 (layer of top component ('h',8,7)).

Learn an r-dim orthonormal subspace B (QR-parameterized, r in {1,4,16}) of the
residual stream entering block 8, such that the interchange intervention on the
CORRUPTED run
    x[:, pos] += B B^T (x_clean[:, pos] - x[:, pos])
makes the model predict the SOURCE (clean) sequence's successor digit.
Sites tried: pos = final position [7], and digit positions [0,4].
Train on 30 pairs (CE toward source answer at final position); evaluate held-out
10 pairs: flip rate to source answer + recovered margin fraction, vs a random
orthonormal subspace control (5 seeds) and the full-residual patch upper bound.
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
S = torch.load(f'{OUT}/stimuli.pt')
clean, corr = S['clean'].cuda(), S['corr'].cuda()
ca, xa = S['clean_ans'].cuda(), S['corr_ans'].cuda()
TRAIN = slice(0, 30); HELD = slice(30, 40)

# cache clean residual entering block 8 for all 40 pairs (chunks of 8)
clean_resid = []
with torch.no_grad():
    for i in range(0, 40, 8):
        cch = {}
        forward(m, clean[i:i+8], cache=cch)
        clean_resid.append(cch[('resid_a', LAYER)])
clean_resid = torch.cat(clean_resid, 0)  # [40, 8, D]


def hook_factory(Bmat, pos, src_resid):
    def hook(site, li, x):
        if site == 'pre_attn' and li == LAYER:
            x = x.clone()
            delta = src_resid[:, pos] - x[:, pos]            # [b, |pos|, D]
            x[:, pos] = x[:, pos] + delta @ Bmat @ Bmat.T
        return x
    return hook


def evaluate(Bmat, pos, sl):
    """Run corrupted[sl] with interchange; return flip rate to source answer,
    recovered margin fraction, on that slice."""
    flips, Mp, Mc, Mx = [], [], [], []
    idxs = range(sl.start, sl.stop, 8)
    with torch.no_grad():
        for i in idxs:
            j = min(i + 8, sl.stop)
            src = clean_resid[i:j]
            lg = forward(m, corr[i:j], resid_hook=hook_factory(Bmat, pos, src))[:, -1].float()
            lgc = forward(m, clean[i:j])[:, -1].float()
            lgx = forward(m, corr[i:j])[:, -1].float()
            n = torch.arange(j - i, device='cuda')
            flips.append((lg.argmax(-1) == ca[i:j]).float())
            Mp.append(lg[n, ca[i:j]] - lg[n, xa[i:j]])
            Mc.append(lgc[n, ca[i:j]] - lgc[n, xa[i:j]])
            Mx.append(lgx[n, ca[i:j]] - lgx[n, xa[i:j]])
    flips = torch.cat(flips); Mp = torch.cat(Mp); Mc = torch.cat(Mc); Mx = torch.cat(Mx)
    rf = ((Mp - Mx) / (Mc - Mx)).mean().item()
    return flips.mean().item(), rf


def train_subspace(r, pos, steps=120, lr=5e-3, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = torch.randn(D, r, generator=g).cuda() * 0.02
    P.requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    order = list(range(0, 30, 8))
    for step in range(steps):
        i = order[step % len(order)]
        j = min(i + 8, 30)
        Bmat, _ = torch.linalg.qr(P)
        src = clean_resid[i:j]
        lg = forward(m, corr[i:j], resid_hook=hook_factory(Bmat, pos, src))[:, -1]
        loss = F.cross_entropy(lg.float(), ca[i:j])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Bmat, _ = torch.linalg.qr(P)
    return Bmat.detach()


res = {}
for pos_name, pos in [('final', [7]), ('digits', [0, 4])]:
    # full-residual upper bound
    full_B = torch.eye(D).cuda()
    fh, rh = evaluate(full_B, pos, HELD)
    res[f'{pos_name}_fullresid'] = {'held_flip': round(fh, 3), 'held_rf': round(rh, 4)}
    print(f'[{pos_name}] FULL residual patch: held flip {fh:.3f}  rf {rh:.4f}', flush=True)
    for r in [1, 4, 16]:
        Bmat = train_subspace(r, pos)
        ftr, rtr = evaluate(Bmat, pos, TRAIN)
        fhe, rhe = evaluate(Bmat, pos, HELD)
        # random-subspace control, 5 seeds
        rf_flips, rf_rfs = [], []
        for sd in range(5):
            g = torch.Generator(device='cpu').manual_seed(100 + sd)
            R, _ = torch.linalg.qr(torch.randn(D, r, generator=g).cuda())
            f0, r0 = evaluate(R, pos, HELD)
            rf_flips.append(f0); rf_rfs.append(r0)
        ctrl_f = sum(rf_flips) / 5; ctrl_r = sum(rf_rfs) / 5
        res[f'{pos_name}_r{r}'] = {
            'train_flip': round(ftr, 3), 'train_rf': round(rtr, 4),
            'held_flip': round(fhe, 3), 'held_rf': round(rhe, 4),
            'random_ctrl_held_flip': round(ctrl_f, 3), 'random_ctrl_held_rf': round(ctrl_r, 4)}
        print(f'[{pos_name}] r={r}: train flip {ftr:.3f}/rf {rtr:.3f} | '
              f'HELD flip {fhe:.3f}/rf {rhe:.3f} | random ctrl flip {ctrl_f:.3f}/rf {ctrl_r:.4f}', flush=True)

json.dump(res, open(f'{OUT}/s3_das.json', 'w'), indent=2)
print('saved s3_das.json')
