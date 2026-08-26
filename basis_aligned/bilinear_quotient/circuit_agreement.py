# circuit_agreement: DOES THE COPULA CIRCUIT CARRY SUBJECT-VERB AGREEMENT? (The
# {11.3, 15.5} ensemble is 17.8x selective for is/was/are prediction — but is it
# choosing WHICH copula (number agreement) or just raising copula probability?)
# Metric (new type): argmax ACCURACY restricted to {is, are} at positions whose
# target is one of them — does the model pick the right number, clean vs removal?
#
# Registered predictions:
#   pred_a clean is/are restricted accuracy >= .75 (the model does agreement).
#   pred_b removing {11.3, 15.5} drops restricted accuracy by >= .10.
#   pred_c global CE rise stays <= .01 (the damage is agreement-shaped, not broad).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_agreement_results.json'
NR = 1920
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
ENSEMBLE = [(11, 3), (15, 5)]
HOOK = {'on': False}


def mk_hook(L):
    def hook(mod, args):
        if not HOOK['on']:
            return None
        hs = [hh for (LL, hh) in ENSEMBLE if LL == L]
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    tid_is = ENC.encode(' is')[0]
    tid_are = ENC.encode(' are')[0]
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L, _ in ENSEMBLE]

    def measure(on):
        HOOK['on'] = on
        gs = 0.0; gn = 0; correct = 0; total = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk].sum()); gn += int(mk.sum())
            cls = ((tg == tid_is) | (tg == tid_are)) & mk
            if int(cls.sum()):
                li = lo[..., tid_is][cls]
                la = lo[..., tid_are][cls]
                pred_is = li > la
                truth_is = (tg[cls] == tid_is)
                correct += int((pred_is == truth_is).sum())
                total += int(cls.sum())
        HOOK['on'] = False
        return gs / max(gn, 1), correct / max(total, 1), total

    g0, acc0, n = measure(False)
    g1, acc1, _ = measure(True)
    for hk in hooks:
        hk.remove()
    pa = acc0 >= 0.75
    pb = (acc0 - acc1) >= 0.10
    pc = (g1 - g0) <= 0.01
    out = {'clean': {'global_ce': round(g0, 4), 'isare_acc': round(acc0, 4),
                     'n_positions': n},
           'removed': {'global_ce': round(g1, 4), 'isare_acc': round(acc1, 4)},
           'acc_drop': round(acc0 - acc1, 4), 'global_rise': round(g1 - g0, 4),
           'pred_a_clean_75': bool(pa), 'pred_b_drop_10': bool(pb),
           'pred_c_global_le_01': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
