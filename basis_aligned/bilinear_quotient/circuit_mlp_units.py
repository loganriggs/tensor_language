# circuit_mlp_units: DO CIRCUITS EXTEND INTO MLP UNITS? (User question: circuits so
# far are attention heads only; and does the top-K-unit compression HELP circuits?)
# For four classes (newline, comma, question, pronouns): rank every deep MLP's
# hidden units by u_class-alignment x unit std (score_u = (u_class . Down_col_u) x
# std(h_u)); pick the single best-scoring MLP; ablate its top-64 class-ranked units
# (their contribution replaced by its mean over fit rows); grade class vs global CE
# rise, against a random-64-unit control in the same MLP.
#
# Registered predictions:
#   pred_a >= 2 of 4 classes yield MLP-unit ablation selectivity >= 2x.
#   pred_b class-ranked units beat random-64 units on class rise by >= 5x at those.
#   pred_c the implicated MLP is layer >= 12 (late, near the announcers).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_mlp_units_results.json'
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
            sc = (u @ H[L].mlp.Down.weight.float()).abs() * SD[L]
            tot = float(sc.topk(64).values.sum())
            if tot > best[1]:
                best = (L, tot, sc)
        L, _, sc = best
        topu = sc.argsort(descending=True)[:64]
        mean_contrib = (MU[L][topu] @ H[L].mlp.Down.weight.float()[:, topu].T)
        g0, c0 = measure(mask_v)
        ABL.update({'L': L, 'topu': topu, 'mean_contrib': mean_contrib})
        g1, c1 = measure(mask_v)
        g_rand = torch.Generator().manual_seed(7)
        rnd = torch.randperm(len(SD[L]), generator=g_rand)[:64].to(DEV)
        ABL.update({'topu': rnd,
                    'mean_contrib': (MU[L][rnd]
                                     @ H[L].mlp.Down.weight.float()[:, rnd].T)})
        g2, c2 = measure(mask_v)
        ABL['L'] = None
        sel = (c1 - c0) / max(g1 - g0, 1e-6)
        ratio = (c1 - c0) / max(c2 - c0, 1e-6)
        sels[cname] = sel; ratios[cname] = ratio; layers_used[cname] = L
        res[cname] = {'mlp': L, 'rise_class': round(c1 - c0, 4),
                      'rise_global': round(g1 - g0, 4),
                      'selectivity': round(sel, 2),
                      'random64_class_rise': round(c2 - c0, 4),
                      'vs_random': round(ratio, 1)}
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    good = [cn for cn in CLS if sels[cn] >= 2]
    pa = len(good) >= 2
    pb = all(ratios[cn] >= 5 for cn in good) if good else False
    pc = all(layers_used[cn] >= 12 for cn in good) if good else False
    out = {'res': res, 'pred_a_2of4_selective': bool(pa),
           'pred_b_beats_random_5x': bool(pb), 'pred_c_late_layers': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
