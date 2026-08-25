# digit_copy_split: IS THE DIGIT CAPABILITY SUBSTANTIALLY COPYING? §1353 found the a8
# digit service is a redundant pair 8.3+8.7, and 8.3 is the §1207-09 COPY STATION. Split
# digit targets by COPY SUPPORT (same digit token within 128 back) and decompose the
# damage per side for: 8.3 solo, 8.7 solo, the pair, and the full layer.
#
# Registered predictions:
#   pred_a 8.3 IS COPY-CONCENTRATED: >= 60% of its total digit damage lies on the
#          copy-supported side.
#   pred_b DIVISION OF LABOR: 8.7's fresh-side share exceeds 8.3's by >= 0.15.
#   pred_c COPYABLE DIGITS ARE EASY DIGITS: copy-supported base CE >= 1.0 nat below fresh.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digit_copy_split_results.json'
NMEAN = 24; NR = 1920
L13 = 8
H = m.transformer.h
LAYERS = (13,)
CUR = {'heads': None, 'mean': None}       # heads: set of head idx to ABLATE


def cproj_hook(mod, args):
    if CUR['heads'] is None:
        return None
    y = args[0].clone()
    for h in CUR['heads']:
        y[..., h * 128:(h + 1) * 128] = CUR['mean'][h].to(y.dtype)
    return (y,)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    dg = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if d.strip().isdigit():
            dg.add(tok)
    dg_ids = torch.tensor(sorted(dg))
    print(f"digit ids {len(dg)}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-head y means at L13's c_proj input from MEANR
    caps = []
    hk = H[L13].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().reshape(-1, 9, 128).mean(0)))
    CUR['heads'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    CUR['mean'] = torch.stack(caps).mean(0)

    # target mask: next tok is pure digits; split by copy support (same token <=128 back)
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    TARGET = torch.isin(tgt_all, dg_ids)
    TARGET[:, :64] = False
    COPY = torch.zeros_like(TARGET)
    # for each position p: does tgt_all[p] appear in toks[p-127..p]?
    B2, T2 = toks.shape
    for r in range(B2):
        row = toks[r].tolist(); tgtrow = tgt_all[r].tolist()
        recent = {}
        for p in range(T2):
            t = tgtrow[p]
            if TARGET[r, p] and t in recent and p - recent[t] <= 128:
                COPY[r, p] = True
            recent[row[p]] = p
    CTAR = TARGET & COPY
    FTAR = TARGET & ~COPY
    print(f"copy-supported {int(CTAR.sum())} | fresh {int(FTAR.sum())}", flush=True)
    # jitter control: target positions shifted +2 (clamped), excluding real targets
    JIT = torch.zeros_like(TARGET)
    JIT[:, 2:] = TARGET[:, :-2]
    JIT &= ~TARGET
    # random control: count-matched draw from non-target positions
    g = torch.Generator().manual_seed(97)
    scores = torch.rand(TARGET.shape, generator=g)
    scores[TARGET | JIT] = -1.0; scores[:, :64] = -1.0
    k = int(TARGET.sum())
    flat = scores.flatten()
    idx_top = flat.topk(k).indices
    RAND = torch.zeros_like(flat, dtype=torch.bool); RAND[idx_top] = True
    RAND = RAND.view(TARGET.shape)
    ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
    print(f"targets {k} | jitter {int(JIT.sum())} | rand {int(RAND.sum())}", flush=True)

    hooks = [H[L13].attn.c_proj.register_forward_pre_hook(cproj_hook)]

    def ce_sides(abl):
        CUR['heads'] = abl
        sc = fc = 0.0; nc = nf = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mc = CTAR[i:i + 8].to(DEV); mf = FTAR[i:i + 8].to(DEV)
            sc += float(ce[mc].sum()); nc += int(mc.sum())
            fc += float(ce[mf].sum()); nf += int(mf.sum())
        return sc / max(nc, 1), fc / max(nf, 1)

    res = {}
    for name, abl in (('base', None), ('h3', {3}), ('h7', {7}), ('pair', {3, 7}),
                      ('layer', set(range(9)))):
        c, f = ce_sides(abl)
        res[name] = {'copy': round(c, 4), 'fresh': round(f, 4)}
        print(f"{name}: copy {c:.4f} | fresh {f:.4f}", flush=True)
    for h in hooks:
        h.remove()

    dmg = {k: {'copy': round(res[k]['copy'] - res['base']['copy'], 4),
               'fresh': round(res[k]['fresh'] - res['base']['fresh'], 4)}
           for k in res if k != 'base'}
    nC = int(CTAR.sum()); nF = int(FTAR.sum())
    tot3 = dmg['h3']['copy'] * nC + dmg['h3']['fresh'] * nF
    tot7 = dmg['h7']['copy'] * nC + dmg['h7']['fresh'] * nF
    copyfrac3 = dmg['h3']['copy'] * nC / max(tot3, 1e-6)
    copyfrac7 = dmg['h7']['copy'] * nC / max(tot7, 1e-6)
    pa = copyfrac3 >= 0.60
    pb = (1 - copyfrac7) - (1 - copyfrac3) >= 0.15
    pc = res['base']['fresh'] - res['base']['copy'] >= 1.0
    out = {'n_copy': nC, 'n_fresh': nF, 'ce': res, 'damage': dmg,
           'copyfrac_83': round(copyfrac3, 4), 'copyfrac_87': round(copyfrac7, 4),
           'pred_a_83_copy_concentrated': bool(pa),
           'pred_b_division_of_labor': bool(pb),
           'pred_c_copy_digits_easy': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ncopyfrac 8.3 {copyfrac3:.3f} | 8.7 {copyfrac7:.3f} | base copy {res['base']['copy']} fresh {res['base']['fresh']}")
    print(f"pred_a copy-conc {pa} | pred_b division {pb} | pred_c easy {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
