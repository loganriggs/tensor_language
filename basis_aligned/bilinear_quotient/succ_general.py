# succ_general: IS 8.7 A GENERAL SUCCESSOR HEAD? (§1277 follow-on.) Digits: 8/8 rank-1 in
# pure weights. Same weights instrument on WEEKDAYS (Monday->Tuesday...) and MONTHS
# (January->February...), plus a behavioural cell where corpus frequency allows.
#
# Registered predictions:
#   pred_a WEIGHTS GENERALIZE: 8.7's value map ranks W+1 top-3 among the 7 weekdays for
#          >= 4 of 6 (non-cyclic) pairs AND M+1 top-3 among the 12 months for >= 6 of 11.
#   pred_b CONTROL FLAT: head 8.1 achieves <= 2 (weekdays) and <= 4 (months).
#   pred_c BEHAVIOURAL (evaluable only if >= 30 targets in 960 rows): 8.7 mean-ablation
#          concentration >= 3 at weekday/month-successor targets; else logged UNEVALUABLE
#          per the score_bar rule and the weights half stands alone.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'succ_general_results.json'
NR = 960
H = m.transformer.h
are = sys.modules[type(H[0].attn).__module__].apply_rotary_emb

WDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')

    def lex_ids(words):
        out = []
        for w in words:
            ids = set()
            for form in (w, ' ' + w):
                t = enc.encode(form)
                if len(t) == 1:
                    ids.add(t[0])
            out.append(torch.tensor(sorted(ids), device=DEV))
        return out

    wd = lex_ids(WDAYS)
    mo = lex_ids(MONTHS)
    print(f"weekday id counts {[len(x) for x in wd]} | month {[len(x) for x in mo]}", flush=True)

    at8 = H[8].attn; at0 = H[0].attn
    lam = float(at8.lamb)
    W_u = m.lm_head.weight.float()

    def head_map(h, toks):
        x = F.rms_norm(m.transformer.wte(toks), (D,))
        v8 = at8.c_v(x).view(-1, 9, 128)[:, h]
        v0 = at0.c_v(x).view(-1, 9, 128)[:, h]
        vv = (1 - lam) * v8 + lam * v0
        y = torch.zeros(vv.shape[0], 9, 128, device=DEV, dtype=vv.dtype)
        y[:, h] = vv
        return (at8.c_proj(y.reshape(-1, D)).float() @ W_u.T)

    def eval_lex(h, lex, ncyc):
        """lex: list of id-tensors; ncyc pairs (i -> i+1), i in 0..ncyc-1. Returns top3 count + ranks."""
        top3 = 0; ranks = []
        for i in range(ncyc):
            if len(lex[i]) == 0 or len(lex[i + 1]) == 0:
                ranks.append(None); continue
            lg = head_map(h, lex[i]).mean(0)
            set_means = [float(lg[lex[j]].mean()) if len(lex[j]) else -1e9 for j in range(len(lex))]
            succ = set_means[i + 1]
            order = sorted(set_means, reverse=True)
            rank = order.index(succ) + 1
            ranks.append(rank)
            top3 += int(rank <= 3)
        return top3, ranks

    w87_t3, w87_r = eval_lex(7, wd, 6)
    m87_t3, m87_r = eval_lex(7, mo, 11)
    w81_t3, _ = eval_lex(1, wd, 6)
    m81_t3, _ = eval_lex(1, mo, 11)
    print(f"8.7 weekdays top3 {w87_t3}/6 ranks {w87_r}", flush=True)
    print(f"8.7 months top3 {m87_t3}/11 ranks {m87_r}", flush=True)
    print(f"8.1 weekdays {w81_t3}/6 | months {m81_t3}/11", flush=True)

    # behavioural cell
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    tgt_all = ROWS[:, 1:]
    TGT = torch.zeros_like(tgt_all, dtype=torch.bool)
    for lex, ncyc in ((wd, 6), (mo, 11)):
        for i in range(ncyc):
            if len(lex[i]) == 0 or len(lex[i + 1]) == 0:
                continue
            is_s = torch.isin(tgt_all, lex[i + 1].cpu())
            prev = torch.isin(ROWS[:, :-1], lex[i].cpu())
            ctx = torch.zeros_like(prev)
            for w in range(1, 129):
                sh = torch.zeros_like(prev)
                sh[:, w:] = prev[:, :-w]
                ctx |= sh
            TGT |= (is_s & ctx)
    TGT[:, :64] = False
    ntar = int(TGT.sum())
    print(f"behavioural successor targets: {ntar}", flush=True)

    beh = {'n_targets': ntar, 'evaluable': ntar >= 30}
    if ntar >= 30:
        ys = []
        def cproj_hook(mod, args):
            ys.append(args[0].detach().float().mean((0, 1)))
        hh = at8.c_proj.register_forward_pre_hook(cproj_hook)
        def fwd(idx):
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        for i in range(0, 24, 4):
            fwd(ROWS[i:i + 4, :-1].to(DEV).contiguous())
        hh.remove()
        ymean = torch.stack(ys).mean(0)
        HSEL = {'on': False}
        def abl_hook(mod, args):
            if HSEL['on']:
                y = args[0].clone()
                y[:, :, 7 * 128:8 * 128] = ymean[7 * 128:8 * 128].to(y.dtype)
                return (y,)
            return args
        hh = at8.c_proj.register_forward_pre_hook(abl_hook)
        ELSE = ~TGT; ELSE[:, :64] = False
        def ce_sets(on):
            HSEL['on'] = on
            tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
            for i in range(0, NR, 8):
                bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
                for name, mask in (('t', TGT), ('e', ELSE)):
                    mm = mask[i:i + 8].to(DEV)
                    tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
            return {k: tots[k] / max(ns[k], 1) for k in tots}
        b = ce_sets(False); a = ce_sets(True)
        hh.remove()
        dt = a['t'] - b['t']; de = a['e'] - b['e']
        beh.update({'dmg_target': round(dt, 4), 'dmg_else': round(de, 5),
                    'conc': round(dt / max(de, 1e-4), 2)})
        print(f"behavioural: dmg {dt:.4f} vs else {de:.5f} conc {beh['conc']}", flush=True)

    pa = w87_t3 >= 4 and m87_t3 >= 6
    pb = w81_t3 <= 2 and m81_t3 <= 4
    pc = (beh.get('conc', 0) >= 3) if beh['evaluable'] else None
    out = {'weights': {'w87_top3': w87_t3, 'w87_ranks': w87_r, 'm87_top3': m87_t3, 'm87_ranks': m87_r,
                       'w81_top3': w81_t3, 'm81_top3': m81_t3},
           'behavioural': beh,
           'pred_a_generalizes': bool(pa), 'pred_b_control': bool(pb),
           'pred_c_behavioural': pc if pc is None else bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
