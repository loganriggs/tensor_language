# circuit_h124: IS HEAD 12.4 A COREFERENCE HEAD? (S1556: it alone carries the
# pronoun circuit at 275x.) Removal of 12.4 scored on: pronoun targets WITH a
# capitalized token in the previous 64 context tokens (antecedent present) vs
# WITHOUT; per-pronoun breakdown (he/she/they); global.
# Registered:
#   pred_a global rise <= .005 (surgical).
#   pred_b antecedent-present pronouns hit >= 2x antecedent-absent (the head
#          tracks discourse entities, not just pronoun frequency).
#   pred_c damage spread across he/she/they within 3x (one mechanism, all
#          pronouns).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken
import re

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_h124_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
C126 = CONSTS['head12.4'].to(DEV).float()
HOOK = {'on': False}


def h126_hook(mod, args):
    if HOOK['on']:
        x = args[0].clone()
        x[:, :, 4 * 128:5 * 128] = C126.to(x.dtype)
        return (x,)
    return None


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    NR = 1920
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    PR = {}
    for w in ('he', 'she', 'they'):
        ids = [ENC.encode(f' {w}')[0], ENC.encode(f' {w.capitalize()}')[0]]
        PR[w] = torch.tensor(ids)
    ALLP = torch.cat(list(PR.values()))
    CAPV = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(r'^ [A-Z]', ENC.decode([t])):
            CAPV[t] = True
    hk = H[12].attn.c_proj.register_forward_pre_hook(h126_hook)

    def measure(on):
        HOOK['on'] = on
        gs = 0.0; gn = 0
        buckets = {k: [0.0, 0] for k in
                   ('ante', 'no_ante', 'he', 'she', 'they')}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk].sum()); gn += int(mk.sum())
            isp = torch.isin(tg.cpu(), ALLP).to(DEV) & mk
            hascap = torch.zeros_like(isp)
            capm = CAPV.to(DEV)[idx]
            cum = torch.cumsum(capm.float(), dim=1)
            past64 = cum - torch.roll(cum, 64, dims=1)
            past64[:, :64] = cum[:, :64]
            hascap = past64 > 0
            for nm, msk in (('ante', isp & hascap), ('no_ante', isp & ~hascap)):
                buckets[nm][0] += float(ce[msk].sum())
                buckets[nm][1] += int(msk.sum())
            for w in ('he', 'she', 'they'):
                msk = torch.isin(tg.cpu(), PR[w]).to(DEV) & mk
                buckets[w][0] += float(ce[msk].sum())
                buckets[w][1] += int(msk.sum())
        HOOK['on'] = False
        return gs / max(gn, 1), {k: v[0] / max(v[1], 1) for k, v in
                                 buckets.items()}, \
            {k: v[1] for k, v in buckets.items()}

    g0, b0, nn = measure(False)
    g1, b1, _ = measure(True)
    hk.remove()
    rises = {k: round(b1[k] - b0[k], 4) for k in b0}
    gl = round(g1 - g0, 4)
    print('global', gl, 'rises', rises, 'n', nn)
    pa = gl <= 0.005
    pb = rises['ante'] >= 2 * max(rises['no_ante'], 1e-6)
    prs = [max(rises[w], 1e-6) for w in ('he', 'she', 'they')]
    pc = max(prs) / min(prs) <= 3
    out = {'global_rise': gl, 'rises': rises, 'n_positions': nn,
           'pred_a_global_005': bool(pa), 'pred_b_ante_2x': bool(pb),
           'pred_c_even_3x': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
