# DOES THE PROGRAM TRACK THE MODEL, OR ONLY ITS AVERAGE LOSS?
#
# Everything certified in `_CONTEXT_FREE_TABLE_FRONTIER` and `_POSITION_WISE_CLASS_CEILING` is a
# CROSS-ENTROPY figure. CE is an average over the tokens the data happens to sample, and two
# predictors can share a CE while choosing entirely different next tokens and while disagreeing wildly
# on the mass they put where the data never looks. Nothing in §1747-§1787 has checked either.
#
# Three instruments the thread has never used, on the settled standalone program (context-free tables,
# output-NN fallback with a rank-64 embedding->row map):
#   TOP-1 AGREEMENT   how often the program's argmax equals the LIVE model's argmax
#   TOP-1 ACCURACY    how often each one's argmax equals the TRUE next token
#   KL(live || prog)  the distributional distance, which unlike the CE gap is an expectation under
#                     the model rather than under the sampled targets
#
# This is a second-class confirmation with a DIFFERENT instrument, not a replication of the same one.
# It can only strengthen or break claims that are already certified on CE.
#
# ROLES. skip7000, skip11000, skip1200; covered and all-position. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, with margins per LESSON 40, each read back against
# its own sentence per LESSON 39:
#   pred_a AGREEMENT IS LOW: the full-rank program's top-1 agreement with the live model is below 50%
#          on all positions at every role. The program is 3 nats of CE away from the model, so it
#          should not be choosing the same token most of the time. If FALSE -- if it agrees with the
#          model more than half the time -- then a per-token program reproduces the model's actual
#          choices far better than its CE suggests, and that is a much stronger claim than anything
#          in this thread.
#   pred_b IT IS STILL A REAL PREDICTOR: the full-rank program's top-1 accuracy against the TRUE next
#          token is at least half the live model's, at every role. Scored independently of pred_a. If
#          FALSE the program is much worse at the actual task than its CE implies, and every recovery
#          figure in §1747-§1787 overstates what it does.
#   pred_c KL EXCEEDS THE CE GAP BY AT LEAST 2x on covered positions at every role. The CE gap is an
#          expectation over sampled targets; KL is an expectation over the model's own distribution
#          and sees the mass the data never visits. If FALSE the program is close to the model
#          distributionally and not merely on the tokens that happen to occur -- which would be the
#          more favourable outcome and worth registering as such.
#   pred_d CONTROLS: CE reproduces §1787's settled arms -- 6.01167 / 5.98477 / 6.00165 at full rank and
#          6.17330 / 6.15261 / 6.14463 at table rank 64 -- within 0.002; live covered CE reproduces
#          3.29205 / 3.09711 / 3.40277; coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 64)
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/program_agreement_beyond_ce_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1787 = {'full': {'skip7000': 6.01167, 'skip11000': 5.98477, 'skip1200': 6.00165},
         '64': {'skip7000': 6.17330, 'skip11000': 6.15261, 'skip1200': 6.14463}}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def forward_logits(idx, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    STATE['idx'] = idx
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def compare(rows, hooks):
    """Live and program on the same batch: CE, top-1 agreement, top-1 accuracy, KL(live||prog)."""
    a = {p: {'ce_l': 0.0, 'ce_p': 0.0, 'agree': 0, 'acc_l': 0, 'acc_p': 0, 'kl': 0.0, 'n': 0}
         for p in ('cov', 'all')}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        ll = forward_logits(idx)[:, 64:].float()
        lp = forward_logits(idx, hooks)[:, 64:].float()
        L = torch.log_softmax(ll, -1)
        P = torch.log_softmax(lp, -1)
        cov = COV['seen'][idx[:, 64:]]
        kl = (L.exp() * (L - P)).sum(-1)
        el = F.cross_entropy(ll.reshape(-1, W), tg.reshape(-1), reduction='none').reshape(tg.shape)
        ep = F.cross_entropy(lp.reshape(-1, W), tg.reshape(-1), reduction='none').reshape(tg.shape)
        al, ap = L.argmax(-1), P.argmax(-1)
        for p, msk in (('cov', cov), ('all', torch.ones_like(cov))):
            a[p]['ce_l'] += float(el.double()[msk].sum()); a[p]['ce_p'] += float(ep.double()[msk].sum())
            a[p]['kl'] += float(kl.double()[msk].sum())
            a[p]['agree'] += int((al == ap)[msk].sum())
            a[p]['acc_l'] += int((al == tg)[msk].sum()); a[p]['acc_p'] += int((ap == tg)[msk].sum())
            a[p]['n'] += int(msk.sum())
    out = {}
    for p in a:
        n = a[p]['n']
        out[p] = {'ce_live': a[p]['ce_l'] / n, 'ce_prog': a[p]['ce_p'] / n,
                  'kl_live_prog': a[p]['kl'] / n, 'top1_agreement': a[p]['agree'] / n,
                  'top1_acc_live': a[p]['acc_l'] / n, 'top1_acc_prog': a[p]['acc_p'] / n, 'n': n}
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    tk = toks.to(DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'PROGRAM AGREEMENT BEYOND CE | top-1 agreement, top-1 accuracy, KL(live||prog) | '
          f'table ranks {RANKS} | DISCOVERY ONLY', flush=True)

    # the settled fallback: output-NN neighbour (§1780/§1781)
    lpc = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

    tables = {st: torch.zeros(ncov, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][i:i + t.shape[0]] = cap[st]
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    print(f'  built the settled fallback and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    def program_rows(r):
        if r is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp = (U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    res = {}
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        row = {}
        for r in RANKS:
            key = 'full' if r is None else str(r)
            fr = program_rows(r)
            c = compare(ev, [(st, row_hook(fr[st])) for st in sites])
            row[key] = {p: {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in c[p].items()} for p in c}
            del fr
            torch.cuda.empty_cache()
        assert abs(row['full']['cov']['ce_live'] - ce_ref) <= 1e-3, (
            f'{ename} live cov CE {row["full"]["cov"]["ce_live"]:.5f} != {ce_ref}')
        res[ename] = row
        print(f'\n  {ename}:', flush=True)
        for r in RANKS:
            key = 'full' if r is None else str(r)
            for p in ('cov', 'all'):
                x = row[key][p]
                print(f'    table {key:5s} {p:3s}  CE live {x["ce_live"]:.5f} prog {x["ce_prog"]:.5f}'
                      f'  KL {x["kl_live_prog"]:.5f}  agree {x["top1_agreement"]:.2%}  '
                      f'acc live {x["top1_acc_live"]:.2%} prog {x["top1_acc_prog"]:.2%}', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    pa = all(res[e]['full']['all']['top1_agreement'] < 0.50 for e in roles)
    pb = all(res[e]['full']['all']['top1_acc_prog']
             >= 0.5 * res[e]['full']['all']['top1_acc_live'] for e in roles)
    pc = all(res[e]['full']['cov']['kl_live_prog']
             >= 2.0 * (res[e]['full']['cov']['ce_prog'] - res[e]['full']['cov']['ce_live'])
             for e in roles)
    live_ref = {e: c for e, _, c in EVAL_SETS}
    pd = (all(abs(res[e][k]['all']['ce_prog'] - v) <= 0.002
              for k, kv in S1787.items() for e, v in kv.items())
          and all(abs(res[e]['full']['cov']['ce_live'] - live_ref[e]) <= 1e-3 for e in roles)
          and ncov == NCOV)

    print(f'\n  top-1 agreement with the live model is below 50% -> {pa}  ' + '  '.join(
        f'{e} {res[e]["full"]["all"]["top1_agreement"]:.2%}' for e in roles), flush=True)
    print(f'  the program keeps >=half the live top-1 accuracy -> {pb}  ' + '  '.join(
        f'{e} {res[e]["full"]["all"]["top1_acc_prog"]:.2%} vs '
        f'{res[e]["full"]["all"]["top1_acc_live"]:.2%}' for e in roles), flush=True)
    print(f'  KL exceeds the covered CE gap by >=2x -> {pc}  ' + '  '.join(
        f'{e} KL {res[e]["full"]["cov"]["kl_live_prog"]:.4f} vs gap '
        f'{res[e]["full"]["cov"]["ce_prog"] - res[e]["full"]["cov"]["ce_live"]:.4f}'
        for e in roles), flush=True)
    print(f'  CE reproduces §1787 + live CE + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'table_ranks': ['full' if r is None else str(r) for r in RANKS],
                     'map_rank': MAP_RANK,
                     'program': 'context-free tables, output-NN fallback with a rank-64 '
                                'embedding->row map -- the settled design of §1780-§1786',
                     'instruments': 'top-1 agreement with the live model, top-1 accuracy against the '
                                    'true next token, and KL(live || program). All three are NEW to '
                                    'this thread, which has been CE-only since §1747.',
                     'ROLE_NOTE': 'DISCOVERY ONLY; a second-class confirmation with a DIFFERENT '
                                  'instrument, not a replication of the same one.'},
          'results': res,
          'predictions': {'pred_a_agreement_below_half': bool(pa),
                          'pred_b_keeps_half_the_accuracy': bool(pb),
                          'pred_c_kl_exceeds_ce_gap_2x': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
