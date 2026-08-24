# comparative_extraction: THE THREE-GOAL LOOP, CLOSED ON ONE CIRCUIT. The comparative
# circuit is the program's most completely named specialist chain: front attention (a02
# band) writes the class mark and key (§1306), head 8.1 alone fetches it (100.7% of
# layer-8's damage, §1304) with a stream-computed criterion (§1305), and the capability
# is predicting "than" after a non-adjacent comparative (§1303). Goal 3 (weights-read
# generalization) is already banked for this circuit (§1307-08 pattern was the matcher's;
# for 8.1 the §1305 fetch analysis). This script runs goals 1 and 2 IN THE SAME HARNESS:
#
#   GOAL 1 (extract, run standalone): keep {8.1 live + a02 band live}, v1-route through
#     every other head (§1314/§1316 grain: removed heads keep their lambda*v1 term — the
#     block-0 broadcast, nearly free in bits — and mean-replace only their fresh values;
#     patterns stay live everywhere and are themselves window-foldable code, §1161-66).
#     MLPs stay live throughout, as in the whole extraction ladder (the ladder extracts
#     ATTENTION; the MLP ladder prices MLPs separately, §1322-27).
#   GOAL 2 (remove the important part, lose ONLY the capability): the same kept set minus
#     8.1 — inside the extraction, does deleting one head delete the capability?
#
# Conditions (single traced forward, one assignment dict — HARNESS.md rule):
#   full        the model
#   ymean       every head's output slice -> per-head mean (the stake anchor)
#   route       ALL heads v1-routed, none live (the free-in-bits crowd baseline)
#   circ        route + {(8,1)} + a02 band live  <- THE EXTRACTED CIRCUIT
#   circ_no81   circ minus (8,1)                 <- goal-2 surgical removal
#   circ_only81 route + {(8,1)} only             <- is the annotator band needed?
#
# Scored at comparative->than targets (next token = "than", a comparative token 2-20 back;
# §1303 mining) and elsewhere, as recovery of the ymean->full gap.
#
# Registered predictions:
#   pred_a THE EXTRACTION CARRIES THE CAPABILITY: circ target recovery >= 0.60 while
#          route alone <= 0.35 — the named heads, not the crowd route, own the gap.
#   pred_b SURGICAL: removing 8.1 inside the extraction forfeits >= 50% of what the
#          circuit heads bought: (rec_circ - rec_no81) >= 0.5 * (rec_circ - rec_route).
#   pred_c THE ANNOTATOR IS LOAD-BEARING: circ_only81 trails circ by >= 0.15 target
#          recovery (§1305's stream-computed criterion needs the a02 mark).
# Diagnostic, not a bar: elsewhere recovery per condition — circ should sit near route
# (the kept heads are specialists; big elsewhere gains would mean we extracted a
# generalist band, not a circuit). n_targets reported; §1303's construction is rare
# (n=54 at 960 rows), this runs 1920 rows and flags n.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_extraction_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02 = {(L, h) for L in (0, 1, 2) for h in range(9)}
KEEPS = {'full': None, 'ymean': set(), 'route': set(),
         'circ': A02 | {(8, 1)}, 'circ_no81': set(A02), 'circ_only81': {(8, 1)}}
MODES = {'full': 'full', 'ymean': 'ymean', 'route': 'route',
         'circ': 'route', 'circ_no81': 'route', 'circ_only81': 'route'}


@torch.no_grad()
def fwd_route(idx, keep, vmeans, ymeans, mode):
    """mode 'full': normal. 'ymean': non-kept heads' y-slices -> per-head mean.
    'route': non-kept heads keep lambda*v1, fresh values -> per-head mean."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        if mode == 'route':
            vpatch = v.clone()
            for h in range(9):
                if (L, h) not in keep:
                    vpatch[:, :, h] = vmeans[L][h].to(vpatch.dtype)
            vv = (1 - at.lamb) * vpatch + at.lamb * v1.view_as(vpatch)
        else:
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        if mode == 'ymean':
            for h in range(9):
                if (L, h) not in keep:
                    y[:, :, h] = ymeans[L][h].to(y.dtype)
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher',
            'lower', 'faster', 'slower', 'older', 'younger', 'stronger', 'weaker',
            'easier', 'harder', 'longer', 'shorter', 'cheaper', 'richer', 'more', 'less',
            'fewer', 'rather']
    than = set(); comp = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if d.strip().lower() == 'than':
            than.add(tok)
        if d.strip().lower() in COMP:
            comp.add(tok)
    than_t = torch.tensor(sorted(than)); comp_t = torch.tensor(sorted(comp))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-head fresh-value means and y means from MEANR (full model)
    vs = [[[] for _ in range(9)] for _ in range(18)]
    ys = [[[] for _ in range(9)] for _ in range(18)]
    caps = {}
    def mkv(L):
        def h(mod, args, out):
            caps[('vin', L)] = args[0].detach()
            return out
        return h
    # capture via a plain full traced pass, re-deriving v and y per layer
    for i in range(0, NMEAN, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                vs[L][h].append(v[:, :, h].float().mean((0, 1)).cpu())
                ys[L][h].append(y[:, :, h].float().mean((0, 1)).cpu())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    vmeans = [torch.stack([torch.stack(vs[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]
    ymeans = [torch.stack([torch.stack(ys[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]

    # target mask (§1303 construction): next tok = "than", comparative 2-20 back
    tgt_all = EVR[:, 1:]; toks = EVR[:, :-1]
    is_comp = torch.isin(toks, comp_t)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp)
        sh[:, w:] = is_comp[:, :-w]
        ctx |= sh
    TARGET = torch.isin(tgt_all, than_t) & ctx
    TARGET[:, :64] = False
    ELSE = ~TARGET
    ELSE[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar} | else {int(ELSE.sum())}", flush=True)

    def ce_cond(cond):
        keep, mode = KEEPS[cond], MODES[cond]
        st = 0.0; se = 0.0; nt = 0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_route(idx, keep, vmeans, ymeans, mode).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    res = {}
    for cond in ('full', 'ymean', 'route', 'circ', 'circ_no81', 'circ_only81'):
        tce, ece = ce_cond(cond)
        res[cond] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{cond}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gap_t = res['ymean']['target'] - res['full']['target']
    gap_e = res['ymean']['else'] - res['full']['else']
    rec = {c: {'target': round((res['ymean']['target'] - res[c]['target']) / max(gap_t, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[c]['else']) / max(gap_e, 1e-6), 4)}
           for c in res if c != 'ymean'}
    rt, rc, rn, ro = (rec['route']['target'], rec['circ']['target'],
                      rec['circ_no81']['target'], rec['circ_only81']['target'])
    pa = rc >= 0.60 and rt <= 0.35
    pb = (rc - rn) >= 0.5 * (rc - rt)
    pc = (rc - ro) >= 0.15
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'recovery': rec,
           'gap_target': round(gap_t, 4), 'gap_else': round(gap_e, 4),
           'pred_a_extraction_carries': bool(pa), 'pred_b_surgical_81': bool(pb),
           'pred_c_annotator_needed': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrecovery target: route {rt} circ {rc} no81 {rn} only81 {ro}")
    print(f"pred_a carries {pa} | pred_b surgical {pb} | pred_c annotator {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
