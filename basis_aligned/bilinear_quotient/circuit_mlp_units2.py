# circuit_mlp_units2: THE SIGNED VERSION (S1563: |.|-ranking mixed promoters and
# suppressors — ablation IMPROVED the class). Rank units by SIGNED alignment
# (u_class . Down_col) x std; ablate the top-64 PROMOTERS and, separately, the
# top-64 SUPPRESSORS (most negative), same mean-substitution, same classes.
#
# Registered predictions:
#   pred_a promoter ablation RAISES class CE >= .05 at >= 2 of 4 classes
#          (promoters are real circuit members).
#   pred_b suppressor ablation LOWERS class CE >= .05 at >= 2 of 4 (gating
#          confirmed with the right sign).
#   pred_c promoter-ablation selectivity >= 2x at >= 2 of 4 (MLP-unit circuit
#          membership established).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_mlp_units2_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}


def mk_mlp_hook(L):
    def hook(mod, args, output):
        if ABL['L'] != L:
            return None
        z = args[0]
        h = (mod.Left(z).float() * mod.Right(z).float())
        sub = h[:, :, ABL['topu']] @ mod.Down.weight.float()[:, ABL['topu']].T
        return (output.float() - sub + ABL['mean_contrib']).to(output.dtype)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    nl = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if '\n' in ENC.decode([t]):
            nl[t] = True
    CLS = {'newline': nl, 'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'pronouns': rx(r'^ (he|she|they|He|She|They)$')}
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    # unit stats for MLPs 4-17 (one pass)
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L))
                 for L in range(4, 18)]
    acc1 = {L: 0 for L in range(4, 18)}; acc2 = {L: 0 for L in range(4, 18)}
    n0 = 0
    for i in range(0, 96, 8):
        store.clear()
        fwd(FR[i:i + 8, :-1].to(DEV).contiguous())
        for L in range(4, 18):
            zz = store[L][0]
            hh_ = (H[L].mlp.Left(zz).float() * H[L].mlp.Right(zz).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + hh_.sum(0); acc2[L] = acc2[L] + (hh_ * hh_).sum(0)
        n0 += 8 * T
    for hk in pre_hooks:
        hk.remove()
    MU = {L: acc1[L] / n0 for L in range(4, 18)}
    SD = {L: (acc2[L] / n0 - MU[L] ** 2).clamp_min(0).sqrt() for L in range(4, 18)}
    hooks = [H[L].mlp.register_forward_hook(mk_mlp_hook(L)) for L in range(4, 18)]
    print("stats done", flush=True)

    def measure(mask_v):
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn, 1)

    res = {}
    sels = {}; ratios = {}; layers_used = {}
    for cname, mask_v in CLS.items():
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        best = (None, -1)
        for L in range(4, 18):
            sc = (u @ H[L].mlp.Down.weight.float()) * SD[L]
            tot = float(sc.abs().topk(64).values.sum())
            if tot > best[1]:
                best = (L, tot, sc)
        L, _, sc = best
        prom = sc.argsort(descending=True)[:64]
        supp = sc.argsort()[:64]
        Wd = H[L].mlp.Down.weight.float()
        g0, c0 = measure(mask_v)
        ABL.update({'L': L, 'topu': prom,
                    'mean_contrib': MU[L][prom] @ Wd[:, prom].T})
        g1, c1 = measure(mask_v)
        ABL.update({'topu': supp,
                    'mean_contrib': MU[L][supp] @ Wd[:, supp].T})
        g2, c2 = measure(mask_v)
        ABL['L'] = None
        sel = (c1 - c0) / max(g1 - g0, 1e-6)
        sels[cname] = sel; ratios[cname] = c2 - c0; layers_used[cname] = L
        res[cname] = {'mlp': L,
                      'promoter_rise_class': round(c1 - c0, 4),
                      'promoter_rise_global': round(g1 - g0, 4),
                      'promoter_selectivity': round(sel, 2),
                      'suppressor_rise_class': round(c2 - c0, 4),
                      'suppressor_rise_global': round(g2 - g0, 4)}
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    pa = sum(1 for cn in CLS if res[cn]['promoter_rise_class'] >= 0.05) >= 2
    pb = sum(1 for cn in CLS if res[cn]['suppressor_rise_class'] <= -0.05) >= 2
    pc = sum(1 for cn in CLS if sels[cn] >= 2) >= 2
    out = {'res': res, 'pred_a_promoters_2of4': bool(pa),
           'pred_b_suppressors_2of4': bool(pb), 'pred_c_selective_2of4': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
