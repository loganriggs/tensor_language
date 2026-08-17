"""Settle the §22 anomaly: why does replacing layer 16's leader IMPROVE the model?

§22 measured, on pile-10k evaluation data: deleting layer 16's leader costs +0.0337
nats, and the 1,154-parameter surrogate lands 0.0252 nats BELOW the intact baseline
(rank-2: 0.0024 below). A replacement beating the original means the discarded
664k-parameter remainder actively hurts on that data. The registered hypothesis: the
model was trained on fineweb; pile is a distribution shift; the whitened rank-1 core
generalises across the shift while the remainder is fineweb-specific. Truncation as
regularisation.

The hypothesis makes a sharp prediction: ON FINEWEB-LIKE DATA THE IMPROVEMENT
DISAPPEARS (the remainder should help, or at least not hurt, in-distribution). If the
surrogate beats the intact model on fineweb too, the hypothesis is dead and the form is
simply carrying noise (e.g. an undertrained or drifted component).

Method: stream a fresh fineweb sample from the Hub, tokenize with gpt2, and score
   baseline / delete / surrogate / rank-2  on both corpora with identical machinery.
The pile arm re-measures §22's numbers on new pile rows as its own control.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_layer16_battery as B
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction

OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_l16_anomaly_results.json')
LI = 16


def build_fineweb(n_seq=90, T=257):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                      split='train', streaming=True)
    ids, chunks = [], []
    for doc in ds:
        ids.extend(tok(doc['text'])['input_ids'])
        while len(ids) >= T:
            chunks.append(torch.tensor(ids[:T]))
            ids = ids[T:]
            if len(chunks) >= n_seq:
                return torch.stack(chunks)
    return torch.stack(chunks)


def ce_on(tokens, hook):
    B.COEFF_FN = hook
    try:
        tot, n = 0.0, 0
        for i in range(0, tokens.shape[0], 6):
            ce = B.fwd_hook(tokens[i:i + 6].to(DEV))
            tot += float(ce.sum()); n += ce.numel()
        return tot / n
    finally:
        B.COEFF_FN = None


def main():
    t0 = time.time()
    fw_path = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
               'fineweb_eval_tokens.pt')
    import os
    if os.path.exists(fw_path):
        fine = torch.load(fw_path)
    else:
        print('streaming fineweb sample from the Hub...')
        fine = build_fineweb()
        torch.save(fine, fw_path)
    print(f'fineweb eval: {tuple(fine.shape)} | pile eval: fresh rows 452:512')
    pile = FW[452:512, :257]

    # rebuild the battery's surrogate objects deterministically
    Y = B.collect0(FW[0:300, :513])
    _, _, Vh = torch.linalg.svd((Y - Y.mean(0)).float(), full_matrices=False)
    Q = orth(Vh[:32].T)
    phi = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                     'bilin18_layer16_battery_results_phi.pt').mean(1)
    d0 = Q[:, int(phi.argmax())].float()
    mlp = m.transformer.h[LI].mlp
    M = form_for_direction(mlp, d0).float()
    E_l, X_l = [], []
    for i in range(0, 96, 6):
        r_ = B.tracked0(FW[i:i + 6, :513].to(DEV))
        X_l.append(r_[2])
    Xh = torch.cat(X_l)
    c_fit = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
    cbar = float(c_fit.mean())
    S = (Xh.T @ Xh / Xh.shape[0]).double()
    ev, U = torch.linalg.eigh(S)
    kd = ev > 1e-8 * ev.max()
    Sh = (U[:, kd] * ev[kd].sqrt()) @ U[:, kd].T
    Sih = (U[:, kd] * ev[kd].rsqrt()) @ U[:, kd].T
    Mw = Sh @ M.double() @ Sh
    ew, Uw = torch.linalg.eigh(Mw)
    u = (Sih @ Uw[:, ew.abs().argmax()]).float(); u = u / u.norm()
    p2 = (Xh @ u) ** 2
    co = torch.linalg.lstsq(torch.stack([p2, torch.ones_like(p2)], 1),
                            c_fit[:, None]).solution.squeeze()
    a_s, b_s = float(co[0]), float(co[1])
    idx2 = ew.abs().argsort(descending=True)[:2]
    M2 = (Sih @ (Uw[:, idx2] * ew[idx2]) @ Uw[:, idx2].T @ Sih).float()
    b2 = float((c_fit - torch.einsum('ni,ij,nj->n', Xh, M2, Xh)).mean())

    def hook_del(xhat, mo):
        c = mo.float() @ d0
        return mo + ((cbar - c)[..., None] * d0).to(mo.dtype)

    def hook_sur(xhat, mo):
        c = mo.float() @ d0
        chat = a_s * (xhat.float() @ u) ** 2 + b_s
        return mo + ((chat - c)[..., None] * d0).to(mo.dtype)

    def hook_rk2(xhat, mo):
        c = mo.float() @ d0
        xf = xhat.float()
        chat = torch.einsum('...i,ij,...j->...', xf, M2, xf) + b2
        return mo + ((chat - c)[..., None] * d0).to(mo.dtype)

    out = {'corpora': {}}
    print(f"\n  {'corpus':>9} {'baseline':>9} {'delete':>9} {'surrogate':>10} "
          f"{'rank-2':>9}")
    for tag, toks in (('pile', pile), ('fineweb', fine)):
        b_ = ce_on(toks, None)
        d_ = ce_on(toks, hook_del)
        s_ = ce_on(toks, hook_sur)
        r_ = ce_on(toks, hook_rk2)
        out['corpora'][tag] = {'baseline': b_, 'delete': d_, 'surrogate': s_,
                               'rank2': r_, 'surrogate_vs_base': s_ - b_,
                               'rank2_vs_base': r_ - b_}
        print(f"  {tag:>9} {b_:>9.4f} {d_:>9.4f} {s_:>10.4f} {r_:>9.4f}   "
              f"(surrogate-base {s_-b_:+.4f}, rank2-base {r_-b_:+.4f})", flush=True)

    dp = out['corpora']['pile']['surrogate_vs_base']
    df = out['corpora']['fineweb']['surrogate_vs_base']
    if dp < -0.005 and df > -0.005:
        v = ('HYPOTHESIS CONFIRMED: the improvement exists on pile and vanishes on '
             'fineweb -- the discarded remainder is distribution-specific; truncation '
             'acts as regularisation under the shift')
    elif dp < -0.005 and df < -0.005:
        v = ('HYPOTHESIS REFUTED: the surrogate beats the intact model on fineweb '
             'too -- the remainder hurts in-distribution as well; the full form '
             'carries noise, not fineweb-specific signal')
    else:
        v = 'inconclusive: the pile improvement did not replicate on fresh rows'
    out['verdict'] = v
    print(f'\nVERDICT: {v}')
    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
