# sqrd12_closer_heads: THE OWNER QUESTION AT THE SCORE-FUNCTION BOUNDARY (§1376).
# sqrd12's a7 is its concentrated closer layer (+0.402, conc 8.9, rel depth 0.64 —
# matching bilin12). Does the single-owner shape survive the score-function change? The
# §1215 precedent (implementation differs across score functions) makes a crowd verdict
# genuinely live — this is the first place the owner claim could break.
#
# Registered predictions:
#   pred_a an owner exists: top solo head >= 60% of a7's target damage.
#   pred_b knockout and keep-only AGREE on the top head.
#   pred_c surgical: top head's elsewhere damage <= 10% of its target damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from tier2_model import load_elriggs
import census_lib as cl

D = 768; T = 256; NL = 12
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'sqrd12_closer_heads_results.json'
L7 = 7
NH = 6
HD = 128
NMEAN = 24; NR = 1920
m12, cfg = load_elriggs('sqrd12')
H = m12.transformer.h
CUR = {'heads': None, 'mean': None}


def cproj_hook(mod, args):
    if CUR['heads'] is None:
        return None
    y = args[0].clone()
    for h in CUR['heads']:
        y[..., h * HD:(h + 1) * HD] = CUR['mean'][h].to(y.dtype)
    return (y,)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m12.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m12.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(cl.PT + 'census_state_diverse.pt')
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
    V12 = m12.transformer.wte.weight.shape[0]
    ROWS = ROWS.clamp_max(V12 - 1)
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    capsH = []
    hk = H[L7].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: capsH.append(args[0].detach().float().reshape(-1, NH, HD).mean(0)))
    CUR['heads'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to('cuda').contiguous())
    hk.remove()
    CUR['mean'] = torch.stack(capsH).mean(0)

    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    depth = torch.zeros_like(toks)
    dr = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        dr = (dr + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = dr
    TARGET = torch.isin(tgt_all, close_ids) & (depth > 0)
    TARGET[:, :64] = False
    JIT = torch.zeros_like(TARGET)
    JIT[:, 2:] = TARGET[:, :-2]
    JIT &= ~TARGET
    g = torch.Generator().manual_seed(97)
    sc = torch.rand(TARGET.shape, generator=g)
    sc[TARGET | JIT] = -1.0; sc[:, :64] = -1.0
    k = int(TARGET.sum())
    flat = sc.flatten()
    RAND = torch.zeros_like(flat, dtype=torch.bool)
    RAND[flat.topk(k).indices] = True
    RAND = RAND.view(TARGET.shape)
    ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
    print(f"targets {k}", flush=True)

    hooks = [H[L7].attn.c_proj.register_forward_pre_hook(cproj_hook)]

    def ce_all(abl):
        CUR['heads'] = abl
        st = sj = sr = se = 0.0; nt = nj = nr_ = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to('cuda')
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for M, acc in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = acc[i:i + 8].to('cuda')
                s_ = float(ce[mm].sum()); n_ = int(mm.sum())
                if M == 't':
                    st += s_; nt += n_
                elif M == 'j':
                    sj += s_; nj += n_
                elif M == 'r':
                    sr += s_; nr_ += n_
                else:
                    se += s_; ne += n_
        return {'target': st / max(nt, 1), 'jitter': sj / max(nj, 1),
                'random': sr / max(nr_, 1), 'else': se / max(ne, 1)}

    base = ce_all(None)
    full = ce_all(set(range(NH)))
    layer_dmg = full['target'] - base['target']
    print(f"layer dmg {layer_dmg:+.4f} (vs §1376 +0.402)", flush=True)
    solo = {}; keep = {}
    for h in range(NH):
        r = ce_all({h})
        solo[h] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        r2 = ce_all(set(range(NH)) - {h})
        keep[h] = {kk: round(r2[kk] - base[kk], 4) for kk in r2}
        print(f"a7.{h}: solo {solo[h]['target']:+.4f} (else {solo[h]['else']:+.4f}) | "
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
           'top_share': round(solo[top_solo]['target'] / max(layer_dmg, 1e-4), 4),
           'pred_a_owner': bool(pa), 'pred_b_agree': bool(pb),
           'pred_c_surgical': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ntop solo a7.{top_solo} share {out['top_share']} | top keep a7.{top_keep}")
    print(f"pred_a owner {pa} | pred_b agree {pb} | pred_c surgical {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
