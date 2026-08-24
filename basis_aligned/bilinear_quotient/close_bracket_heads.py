# close_bracket_heads: PER-HEAD LOCALIZATION inside a13 (§1338 certified the layer at
# conc 46.8, controls clean). Ablate each of L13's 9 heads SOLO (y-mean of its c_proj
# slice) and each ALL-BUT-ONE (keep head h, ablate the other 8) at bracket-closing
# targets — the §3.1 redundancy trap requires both directions before naming an owner.
#
# Registered predictions:
#   pred_a AN OWNER EXISTS: the top solo head carries >= 60% of the whole-layer target
#          damage (§1338: +0.706). The 46.8x layer concentration and the comparative
#          precedent both point owner-ward; a crowd echoes the question circuit instead.
#   pred_b KNOCKOUT AND KEEP-ONLY AGREE: the same head ranks first by solo damage and
#          by keep-only recovery (all-but-h damage is the LOWEST for the same h).
#   pred_c SURGICAL: the top head's elsewhere damage <= 10% of its target damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_heads_results.json'
NMEAN = 24; NR = 1920
L13 = 13
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
    close_t = set(); open_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
    close_ids = torch.tensor(sorted(close_t)); open_ids = torch.tensor(sorted(open_t))

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

    # target mask: next tok closes AND unmatched open within 64 back
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    depth = torch.zeros_like(toks)
    d_run = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        d_run = (d_run + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = d_run
    TARGET = torch.isin(tgt_all, close_ids) & (depth > 0)
    TARGET[:, :64] = False
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

    def ce_all(abl):
        CUR['heads'] = abl
        outs = {}
        st = sj = sr = se = 0.0; nt = nj = nr_ = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for M, acc in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = acc[i:i + 8].to(DEV)
                s = float(ce[mm].sum()); n = int(mm.sum())
                if M == 't':
                    st += s; nt += n
                elif M == 'j':
                    sj += s; nj += n
                elif M == 'r':
                    sr += s; nr_ += n
                else:
                    se += s; ne += n
        return {'target': st / max(nt, 1), 'jitter': sj / max(nj, 1),
                'random': sr / max(nr_, 1), 'else': se / max(ne, 1)}

    base = ce_all(None)
    full = ce_all(set(range(9)))
    layer_dmg = full['target'] - base['target']
    print(f"layer dmg {layer_dmg:+.4f} (vs §1338 +0.706)", flush=True)
    solo = {}; keep = {}
    for h in range(9):
        r = ce_all({h})
        solo[h] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        r2 = ce_all(set(range(9)) - {h})
        keep[h] = {kk: round(r2[kk] - base[kk], 4) for kk in r2}
        print(f"L13.{h}: solo {solo[h]['target']:+.4f} (else {solo[h]['else']:+.4f}) | "
              f"all-but {keep[h]['target']:+.4f}", flush=True)
        json.dump({'partial': True, 'solo': {str(x): solo[x] for x in solo},
                   'keep': {str(x): keep[x] for x in keep}}, open(OUT, 'w'), indent=1)
    for h in hooks:
        h.remove()

    top_solo = max(solo, key=lambda h: solo[h]['target'])
    top_keep = min(keep, key=lambda h: keep[h]['target'])
    pa = solo[top_solo]['target'] >= 0.60 * max(layer_dmg, 1e-4)
    pb = top_solo == top_keep
    pc = abs(solo[top_solo]['else']) <= 0.10 * max(solo[top_solo]['target'], 1e-4)
    out = {'n_targets': k, 'n_rows': NR, 'layer_dmg': round(layer_dmg, 4),
           'solo': {str(h): solo[h] for h in solo}, 'keep': {str(h): keep[h] for h in keep},
           'top_solo': int(top_solo), 'top_keep': int(top_keep),
           'top_share_of_layer': round(solo[top_solo]['target'] / max(layer_dmg, 1e-4), 4),
           'pred_a_owner': bool(pa), 'pred_b_directions_agree': bool(pb),
           'pred_c_surgical': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ntop solo L13.{top_solo} share {out['top_share_of_layer']} | top keep L13.{top_keep}")
    print(f"pred_a owner {pa} | pred_b agree {pb} | pred_c surgical {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
