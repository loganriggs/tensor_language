# bqlib -- the shared substrate for bilin18 compiled-program experiments.
# BQGATE: LIBRARY  -- no experiment result comes out of this file; see ops/gate.py.
#
# WHY THIS EXISTS. Measured 2026-08-29 across ops/: 219 scripts, 96,769 lines, 89,012 inside functions,
# of which 37,619 (42%) are copies beyond the first. forward_logits appears 173 times, load 214, mk 160,
# row_hook 65, program_rows 45, compare_by_bucket 17. The median gap between one run finishing and the
# next starting is 235s, because every experiment re-emits ~300 known-good lines to change ~40.
# Separately, ~72% of the GPU seconds in the S1935-S1940 lineage went to re-scoring arms a previous run
# had already scored identically, plus a LIVE forward pass every script recomputes on the same eval files.
#
# WHAT IT DOES NOT DO. Per-run setup (model load + 36-table build) was measured at a median 8.3% of
# runtime and ~4s in the fallback lineage. Caching tables ACROSS runs would save essentially nothing, so
# this library does not try; the cache here is on SCORING passes, which is where the recompute is.
#
# CORRECTNESS. Every cached entry stores a fingerprint of the substituted rows it was computed from and
# refuses to load if the fingerprint does not match (PRE-FLIGHT D: a cache must fail loudly, not quietly).
# LIB_VERSION invalidates the whole cache when the build path changes.
#
# USE:
#     import bqlib as B
#     P = B.Program(B.FIT_5419)                 # coverage, 36 tables, neighbour index, map solver
#     am, nll = B.score(P, 'map64', 'skip7000') # per-position top-1 and CE, cached
#     am, nll = B.score(P, None, 'skip7000')    # None == the LIVE model
#     B.report(preds, results, OUT, t0)
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV                                    # noqa: E402

LIB_VERSION = 3          # bump on ANY change to table building, arm construction or forward_logits

D = 1152
T = 256
V = 50257
W = 50304
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
CACHE = PT + '.bqcache/'
H = m.transformer.h
SITES = [(k, L) for k in ('mlp', 'attn') for L in range(18)]

FIT_5419 = PT + '.rowcache/fineweb_n96_skip80.pt'      # 5,419 types at T=256 -- DEPLOYED coverage
FIT_16110 = PT + '.rowcache/fineweb_n480_skip80.pt'    # 16,110 types at T=256 (measured, S1923)
EVAL_SETS = {'skip7000': PT + '.rowcache/fineweb_n192_skip7000.pt',
             'skip11000': PT + '.rowcache/fineweb_n192_skip11000.pt',
             'skip1200': PT + '.rowcache/fineweb_n96_skip1200.pt'}
ROLES = ('skip7000', 'skip11000', 'skip1200')
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
SKIP = 64                # positions >= 64 are scored (house convention)

STATE = {}
STATS = {'hit': 0, 'miss': 0, 'stale': 0, 'unreadable': 0}


# ---------------------------------------------------------------- data + forward

def load(p):
    """A row-cache file as a (docs, T+1) long tensor on CPU."""
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
    """Substitute each position's site output with full_rows[current token] -- the S1765 compilation."""
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
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


# ---------------------------------------------------------------- the program

class Program:
    """Coverage, the 36 context-free tables, the neighbour index, and the map solver, built once.

    arm names:
      'live'                 the unmodified model (pass None to score())
      'nn'                   output-NN neighbour rows for uncovered types (S1780/S1781), ~0.09M
      'map<R>'               rank-R embedding->row map for uncovered types (S1870), 36*R*2*D
      'nn<P>'                cosine-routed: the top P% of uncovered types by neighbour cosine take the
                             neighbour row, the rest take the rank-64 map row (S1939)
      'nn<P>m<R>'            the same, with the routed-out remainder on a rank-R map instead
      'mix<A>m<R>'           BLEND in row space: A% neighbour + (100-A)% rank-R map, every type
      'msk<P>m<R>'           route the top P% of uncovered types by UNC_MASS to the rank-R map,
                             the rest to the neighbour -- S1954's unseen-target router

    table_rank is None (full), an int (uniform), or {'mlp': r, 'attn': r} -- S1928-S1935 found
    MLP-heavy allocation beats uniform for free, with attention wanting 12.5-25% of the
    per-site budget. cost() sums the ACTUAL per-site ranks.
      ('table', R, arm)      the same, with every table truncated to rank R first
    """

    def __init__(self, fit_path, expect_ncov=None, verbose=True):
        t0 = time.time()
        self.fit_path = fit_path
        fit = load(fit_path)
        seen_cpu = torch.zeros(V, dtype=torch.bool)
        seen_cpu[fit[:, :T].reshape(-1).long()] = True
        self.ncov = int(seen_cpu.sum())
        if expect_ncov is not None:
            assert self.ncov == expect_ncov, f'coverage {self.ncov} != {expect_ncov}'
        self.seen = seen_cpu.to(DEV)
        self.tk = seen_cpu.nonzero(as_tuple=True)[0].to(DEV)
        self.unc = (~seen_cpu).nonzero(as_tuple=True)[0].to(DEV)
        self.freq = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(), minlength=V).to(DEV)

        # the 36 context-free tables: each site's output on a length-1 sequence, per covered token
        self.tables = {st: torch.zeros(self.ncov, D, device=DEV) for st in SITES}
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
                return None
            return hook
        for i in range(0, self.ncov, 256):
            t = self.tk[i:i + 256].unsqueeze(1)
            forward_logits(t, [(st, mk(st)) for st in SITES])
            for st in SITES:
                self.tables[st][i:i + t.shape[0]] = cap[st]

        # the output-NN neighbour index AND its cosine (S1780/S1781; the cosine is S1939's router)
        lpc = torch.zeros(self.ncov, W, device=DEV)
        for i in range(0, self.ncov, 256):
            t = self.tk[i:i + 256].unsqueeze(1)
            lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
        pcn = torch.softmax(lpc, -1)
        pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
        del lpc
        self.nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
        self.nnrow[self.tk] = torch.arange(self.ncov, device=DEV)
        self.nnsim = torch.zeros(V, device=DEV)
        # S1954's open question: the blend loses the unseen-target bucket on 5 of 6 role-coverage
        # cells because its neighbour half cannot reach a token no fit row contains. unc_mass is the
        # obvious INPUT-side predictor of that case -- how much probability the live model puts on
        # out-of-table vocabulary from this token alone, on a length-1 sequence. It falls out of the
        # loop that already builds the neighbour index, so it costs nothing.
        self.unc_mass = torch.zeros(V, device=DEV)
        uncmask = torch.zeros(W, dtype=torch.bool, device=DEV)
        uncmask[self.unc] = True
        for s0 in range(0, self.unc.numel(), 512):
            u = self.unc[s0:s0 + 512]
            praw = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
            self.unc_mass[u] = praw[:, uncmask].sum(-1)
            p = praw / praw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            sim, arg = (p.half() @ pcn.T).float().max(-1)
            self.nnrow[u] = arg
            self.nnsim[u] = sim
            del praw
        del pcn
        torch.cuda.empty_cache()

        # the ridge solver for the embedding->row map (S1870)
        self.Ecov = m.transformer.wte.weight.detach()[self.tk].float().double()
        self.A = (self.Ecov.T @ self.Ecov
                  + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (self.ncov / D))
        self.Eunc = m.transformer.wte.weight.detach()[self.unc].float().double()
        self.routefrac = {}
        self._tabmemo = {}
        self._mapmemo = {}
        self.digest = _tables_digest(self)
        self.build_s = time.time() - t0
        if verbose:
            print(f'  [bqlib] coverage {self.ncov} from {os.path.basename(fit_path)}: 36 tables, '
                  f'neighbour index and map solver built ({self.build_s:.0f}s)', flush=True)

    # -------------------------------------------------- arm construction

    @staticmethod
    def _rank_of(table_rank, st):
        """table_rank may be None (full), an int (uniform), or {'mlp': r, 'attn': r} (per-site)."""
        return table_rank[st[0]] if isinstance(table_rank, dict) else table_rank

    MEMO_CAP = 2       # distinct table-rank specs held at once; see _evict()

    def _evict(self):
        """Bound the memos.

        PROFILED 2026-08-29: memoizing the truncated tables and the map SVD per rank spec turned 324
        float64 SVDs per coverage into 36 and was the single biggest speed win (bqlib v2). But the memos
        were UNBOUNDED, and one truncated table set at 16,110 is 36 x 16110 x 1152 x 4 = 2.67 GB. A
        six-point allocation sweep asked for eight specs -- 21 GB of tables plus 6 GB of SVD factors --
        and died with CUDA OOM after the first three arms. The access pattern is one spec at a time, so
        a cap of two keeps essentially all the reuse (score_roles holds one arm across three roles, and
        consecutive arms often share a spec) while bounding memory."""
        while len(self._tabmemo) > self.MEMO_CAP:
            old = next(iter(self._tabmemo))
            del self._tabmemo[old]
            for k in [k for k in self._mapmemo if k[0] == old]:
                del self._mapmemo[k]
            torch.cuda.empty_cache()

    def _tables_at(self, table_rank):
        if table_rank is None:
            return self.tables
        key = _rk_key(table_rank)
        if key in self._tabmemo:
            return self._tabmemo[key]
        out = {}
        for st, tbl in self.tables.items():
            r = self._rank_of(table_rank, st)
            if r is None:
                out[st] = tbl
                continue
            b = tbl.double()
            mu = b.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
            out[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        self._tabmemo[key] = out
        self._evict()
        return out

    def _map(self, tc, st, rank, table_rank=None):
        """Rank-`rank` embedding->row map rows for the uncovered types.

        PROFILED 2026-08-29: 95% of a six-arm two-coverage run (764.5s of 808.0s) was here -- 36 float64
        ridge solves and 36 float64 SVDs of a 1152x1152, recomputed for EVERY arm and EVERY role. The
        solve and the SVD do not depend on `rank` at all; only the truncation does. Memoizing them per
        (table rank, site) turns 324 SVDs per coverage into 36."""
        key = (_rk_key(table_rank), st)
        usv = self._mapmemo.get(key)
        if usv is None:
            Ws = torch.linalg.solve(self.A, self.Ecov.T @ tc[st].double())
            usv = torch.linalg.svd(Ws, full_matrices=False)
            self._mapmemo[key] = usv
            del Ws
        U, S, Vh = usv
        return (self.Eunc @ ((U[:, :rank] * S[:rank]) @ Vh[:rank])).float()

    def arm(self, name, table_rank=None):
        """Full (V, D) substitution rows per site for one arm. `name` as documented on the class."""
        tc = self._tables_at(table_rank)
        out = {}
        if name == 'nn':
            for st in SITES:
                fr = torch.zeros(V, D, device=DEV)
                fr[self.tk] = tc[st]
                fr[self.unc] = tc[st][self.nnrow[self.unc]]
                out[st] = fr
            return out
        if name.startswith('nn') and ('m' in name[2:] or name[2:].isdigit()) and name != 'nn':
            # nn<P>      route the top P% of uncovered types by cosine to the neighbour, rest to map64
            # nn<P>m<R>  same, but the routed-out remainder takes a rank-R map instead
            spec = name[2:]
            pstr, mrank = (spec.split('m', 1) if 'm' in spec else (spec, '64'))
            if not (pstr.isdigit() and mrank.isdigit()):
                raise ValueError(f'unknown arm {name!r}')
            frac = int(pstr) / 100.0
            mrank = int(mrank)
            su = self.nnsim[self.unc]
            tau = torch.quantile(su.double(), 1.0 - frac).float()
            usenn = (su >= tau)
            self.routefrac[name] = float(usenn.float().mean())
            for st in SITES:
                fr = torch.zeros(V, D, device=DEV)
                fr[self.tk] = tc[st]
                fr[self.unc] = torch.where(usenn.unsqueeze(1), tc[st][self.nnrow[self.unc]],
                                           self._map(tc, st, mrank, table_rank))
                out[st] = fr
            return out
        if name.startswith('msk'):
            spec = name[3:]
            pstr, mrank = (spec.split('m', 1) if 'm' in spec else (spec, '512'))
            if not (pstr.isdigit() and mrank.isdigit()):
                raise ValueError(f'unknown arm {name!r}')
            frac, mrank = int(pstr) / 100.0, int(mrank)
            um = self.unc_mass[self.unc]
            tau = torch.quantile(um.double(), 1.0 - frac).float()
            usemap = (um >= tau)
            self.routefrac[name] = float(usemap.float().mean())
            for st in SITES:
                fr = torch.zeros(V, D, device=DEV)
                fr[self.tk] = tc[st]
                fr[self.unc] = torch.where(usemap.unsqueeze(1), self._map(tc, st, mrank, table_rank),
                                           tc[st][self.nnrow[self.unc]])
                out[st] = fr
            return out
        if name.startswith('mix'):
            # mix<A>m<R>: BLEND the two forms in row space, A% neighbour + (100-A)% rank-R map, for
            # every uncovered type. Routing (nn<P>m<R>) gives each token one form or the other;
            # blending gives every token some of both. Different mechanism, same two ingredients.
            spec = name[3:]
            astr, mrank = (spec.split('m', 1) if 'm' in spec else (spec, '64'))
            if not (astr.isdigit() and mrank.isdigit()):
                raise ValueError(f'unknown arm {name!r}')
            al, mrank = int(astr) / 100.0, int(mrank)
            for st in SITES:
                fr = torch.zeros(V, D, device=DEV)
                fr[self.tk] = tc[st]
                fr[self.unc] = (al * tc[st][self.nnrow[self.unc]]
                                + (1.0 - al) * self._map(tc, st, mrank, table_rank))
                out[st] = fr
            return out
        if name.startswith('map'):
            rank = int(name[3:])
            for st in SITES:
                fr = torch.zeros(V, D, device=DEV)
                fr[self.tk] = tc[st]
                fr[self.unc] = self._map(tc, st, rank, table_rank)
                out[st] = fr
            return out
        raise ValueError(f'unknown arm {name!r}')

    def route_fraction(self, name):
        """The fraction of uncovered types a routed arm sends to its first branch.

        Computed directly from the signal, NOT as a side effect of arm(). `routefrac` is populated when
        the rows are built -- so on a warm run, where score() serves every role from cache and arm() is
        never called, a control reading `routefrac` silently reads its default instead of failing. That
        is PRE-FLIGHT D's second direction: the watcher went quiet rather than loud. Found 2026-08-29
        when a fully-cached re-run reported a routed-fraction deviation of exactly 0.5000.
        """
        if name.startswith('msk'):
            frac = int(name[3:].split('m')[0]) / 100.0
            sig = self.unc_mass[self.unc]
        elif name.startswith('nn') and name != 'nn':
            frac = int(name[2:].split('m')[0]) / 100.0
            sig = self.nnsim[self.unc]
        else:
            return None
        tau = torch.quantile(sig.double(), 1.0 - frac).float()
        return float((sig >= tau).float().mean())

    def cost(self, name, table_rank=None):
        """Parameter count of a build, in the S1754 accounting."""
        tab = 0
        for st in SITES:
            r = self._rank_of(table_rank, st)
            tab += (self.ncov * D + D) if r is None else (r * (self.ncov + D) + 2 * D)
        if name == 'nn':
            fb = int(self.unc.numel()) * 2                 # one index per uncovered type
        elif name.startswith('map'):
            fb = 36 * int(name[3:]) * 2 * D
        else:
            spec = name[3:] if name.startswith(('mix', 'msk')) else name[2:]
            mr = int(spec.split('m', 1)[1]) if 'm' in spec else 64
            fb = 36 * mr * 2 * D + int(self.unc.numel()) * 2
        return tab + fb


# ---------------------------------------------------------------- scoring, with a verified cache

def _rk_key(table_rank):
    """A hashable, cache-stable key for a table-rank spec (None / int / per-site dict)."""
    if isinstance(table_rank, dict):
        return tuple(sorted((k, table_rank[k]) for k in table_rank))
    return table_rank


def _tables_digest(prog):
    """Digest of everything an arm is built FROM: the 36 tables, the neighbour index and its cosine.

    Keyed on the Program rather than on constructed rows so that score() can verify a cached entry
    WITHOUT building the arm -- building it was 95% of the runtime (see Program._map)."""
    h = hashlib.sha256()
    for st in SITES:
        r = prog.tables[st]
        h.update(str(tuple(r.shape)).encode())
        h.update(r[::997].float().cpu().numpy().tobytes())
    h.update(prog.nnrow[::97].cpu().numpy().tobytes())
    h.update(prog.nnsim[::97].float().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _fingerprint(prog, armname, table_rank):
    """What a cached entry must match: the program it came from, plus the arm spec."""
    if armname is None:
        return 'live|' + prog.digest[:16]
    return f'{armname}|{_rk_key(table_rank)}|' + prog.digest[:16]


def _key(prog, armname, table_rank, role):
    return hashlib.sha256(
        f'{LIB_VERSION}|{os.path.basename(prog.fit_path)}|{prog.ncov}|{armname}|'
        f'{_rk_key(table_rank)}|{role}|{SKIP}'.encode()).hexdigest()[:24]


@torch.no_grad()
def _run(rows_path_or_none, hooks, role):
    """Per-position top-1 and per-position CE of the true target, over scored positions in fixed order."""
    ev = load(EVAL_SETS[role])
    am, nl = [], []
    for i in range(0, ev.shape[0], 8):
        bb = ev[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, SKIP:]
        lg = forward_logits(idx, hooks)[:, SKIP:]
        am.append(lg.argmax(-1).reshape(-1).to(torch.int32).cpu())
        lp = torch.log_softmax(lg.float(), -1)
        nl.append((-lp.gather(-1, tg.unsqueeze(-1).long()).squeeze(-1)).reshape(-1).float().cpu())
        del lg, lp
    return torch.cat(am), torch.cat(nl)


def cache_path(prog, armname, role, table_rank=None):
    """Where score() would cache this arm-role. Exposed so a validator can corrupt it."""
    return CACHE + _key(prog, 'live' if armname is None else armname, table_rank, role) + '.pt'


def score(prog, armname, role, table_rank=None, use_cache=True, prerows=None):
    """(top1, ce) per scored position for one arm on one role. armname None == the LIVE model.

    Cached on disk under (LIB_VERSION, fit file, coverage, arm, table rank, role). A cached entry
    carries the fingerprint of the rows it was built from and is REJECTED, loudly, on mismatch."""
    name = 'live' if armname is None else armname
    k = _key(prog, name, table_rank, role)
    path = CACHE + k + '.pt'
    fp = _fingerprint(prog, armname, table_rank)
    if use_cache and os.path.exists(path):
        try:
            c = torch.load(path, map_location='cpu')
            if c.get('fp') == fp and c.get('lib') == LIB_VERSION:
                STATS['hit'] += 1
                print(f'  [bqlib] cache HIT  {name:8s} {role}', flush=True)
                return c['am'], c['nl']
            STATS['stale'] += 1
            print(f'  [bqlib] cache STALE {name:8s} {role} -- fingerprint changed, recomputing',
                  flush=True)
        except Exception as e:                                       # never let a bad cache kill a run
            STATS['unreadable'] += 1
            print(f'  [bqlib] cache UNREADABLE {name} {role} ({e}) -- recomputing', flush=True)
    STATS['miss'] += 1
    t0 = time.time()
    rows = prerows if prerows is not None else (None if armname is None
                                                else prog.arm(armname, table_rank))
    hooks = () if rows is None else [(st, row_hook(rows[st])) for st in SITES]
    am, nl = _run(None, hooks, role)
    del rows, hooks
    torch.cuda.empty_cache()
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        tmp = path + '.tmp'
        torch.save({'am': am, 'nl': nl, 'fp': fp, 'lib': LIB_VERSION, 'arm': name, 'role': role}, tmp)
        os.replace(tmp, path)
    print(f'  [bqlib] computed   {name:8s} {role} ({time.time() - t0:.1f}s)', flush=True)
    return am, nl


def score_roles(prog, armname, roles=ROLES, table_rank=None, use_cache=True):
    """{role: (top1, ce)} for one arm, constructing its rows AT MOST ONCE across all roles.

    The per-role score() loop in the first validator built the same arm three times and made a run
    3x SLOWER than the hand-written script it replaced. Prefer this."""
    out, rows = {}, None
    for r in roles:
        if use_cache and os.path.exists(cache_path(prog, armname, r, table_rank)):
            out[r] = score(prog, armname, r, table_rank, use_cache)
            continue
        if rows is None and armname is not None:
            rows = prog.arm(armname, table_rank)
        out[r] = score(prog, armname, r, table_rank, use_cache, prerows=rows)
    del rows
    torch.cuda.empty_cache()
    return out


def axes(prog, role):
    """(true target, input-token-covered mask) per scored position, in score()'s order -- S1936's axis."""
    ev = load(EVAL_SETS[role])
    tg, ic = [], []
    for i in range(0, ev.shape[0], 8):
        bb = ev[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg.append(bb[:, 1:].to(DEV)[:, SKIP:].reshape(-1).cpu())
        ic.append(prog.seen[idx[:, SKIP:]].reshape(-1).cpu())
    return torch.cat(tg), torch.cat(ic)


def ref(results_json, arm, field='ce_prog', cls='pooled', bucket='overall', roles=ROLES):
    """Read a published per-role reference triple from the artifact that produced it.

    Twice on 2026-08-29 I typed a three-role reference triple by hand and got one entry wrong -- once
    by copying the second role's value into the third (S1953), and again ONE SECTION after writing the
    lesson about it (S1956, where the control then failed on my constant rather than on the data).
    A reference that exists in a result JSON should never be retyped. Pass the path the number was
    published from.
    """
    with open(results_json) as fh:
        d = json.load(fh)
    r = d['results']
    if roles[0] not in r:                       # single-coverage runs nest under a coverage key
        r = r[next(iter(r))]
    return tuple(r[role][arm][cls][bucket][field] for role in roles)


def inertness_pairs(plan):
    """Derive the two-sided covered-input control from the PLAN, instead of by hand.

    S1765/S1936: the fallback is consulted only where the current token has no table entry, so two arms
    with the SAME table_rank must be EXACTLY inert at covered inputs, and two with DIFFERENT table_rank
    must move them. Four times on 2026-08-29 I inherited this control across a fork that changed which
    arms differed how, and asserted the wrong polarity -- S1946, S1949, S1951 and again here. Every time
    pred_a/b/c passed 3/3 while pred_d failed on a clause that could not hold. The polarity is a fact
    about the plan, so read it off the plan.

    plan: iterable of (arm, table_rank, label). Returns (must_be_inert, must_differ), each a list of
    (label, label) pairs.
    """
    spec = {lab: _rk_key(tr) for _a, tr, lab in plan}
    labs = [lab for _a, _tr, lab in plan]
    inert, differ = [], []
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            (inert if spec[a] == spec[b] else differ).append((a, b))
    # A two-sided control with an empty side is one-sided and says so. S1957's plan had three arms at
    # three different table ranks, so there were no same-spec pairs and the inertness half had nothing
    # to check -- it reported True vacuously. PRE-FLIGHT D: a watcher that goes quiet looks like a
    # watcher that passed.
    if not inert or not differ:
        which = 'same-spec (inert)' if not inert else 'differing-spec (must move)'
        print(f'  [bqlib] WARNING: no {which} pairs in this plan -- that half of the covered-input '
              f'control is VACUOUS. Add an arm that shares (or differs in) table_rank to restore it.',
              flush=True)
    return inert, differ


def input_ids(prog, role):
    """The INPUT token id at each scored position, in score()'s order.

    axes() returns the target and the input's coverage flag but not the id itself, which is what any
    per-input-token signal (S1954's unc_mass, S1939's cosine) has to be looked up by."""
    ev = load(EVAL_SETS[role])
    out = []
    for i in range(0, ev.shape[0], 8):
        bb = ev[i:i + 8]
        out.append(bb[:, :-1][:, SKIP:].reshape(-1).cpu())
    return torch.cat(out)


def cells(prog, tgt, icov, live, arm):
    """Top-1, kept-fraction and CE per (input-coverage class) x (target bucket), plus pooled.

    live and arm are each an (am, nl) pair from score(). The bucket axis is the TARGET's fit-row
    frequency; the coverage axis is the INPUT token's. They are different axes -- see LESSON 74."""
    freq = prog.freq.cpu()[tgt.long()]
    (lam, lnl), (aam, anl) = live, arm
    o = {}
    for cname, cm in (('covered_input', icov), ('uncovered_input', ~icov),
                      ('pooled', torch.ones_like(icov))):
        o[cname] = {}
        for b in BUCKETS:
            msk = cm & (freq >= b[0]) & (freq <= b[1])
            n = int(msk.sum())
            al = float((lam[msk] == tgt[msk]).float().mean()) if n else 0.0
            ap = float((aam[msk] == tgt[msk]).float().mean()) if n else 0.0
            o[cname][f'{b[0]}-{b[1]}'] = {'n': n, 'top1_acc_live': al, 'top1_acc_prog': ap,
                                          'kept_fraction': ap / max(al, 1e-9),
                                          'ce_live': float(lnl[msk].mean()) if n else 0.0,
                                          'ce_prog': float(anl[msk].mean()) if n else 0.0}
        n = int(cm.sum())
        o[cname]['overall'] = {'n': n,
                               'top1_acc_live': float((lam[cm] == tgt[cm]).float().mean()),
                               'top1_acc_prog': float((aam[cm] == tgt[cm]).float().mean()),
                               'ce_live': float(lnl[cm].mean()),
                               'ce_prog': float(anl[cm].mean())}
        assert (sum(o[cname][f'{x}-{y}']['n'] for x, y in BUCKETS) == n), 'buckets do not partition'
    return o


def paired_t(a, b):
    """Paired per-position difference a - b: mean, standard error, t. The instrument S1939 lacked."""
    dd = (a - b).double()
    n = dd.numel()
    mean = float(dd.mean())
    se = float(dd.std(unbiased=True)) / (n ** 0.5)
    return {'mean': mean, 'se': se, 't': mean / max(se, 1e-12), 'n': n,
            'n_nonzero': int((dd != 0).sum())}


def _write_json(payload, out_path):
    with open(out_path, 'w') as fh:
        fh.write(json.dumps(payload, indent=1))


def report(preds, payload, out_path, t0):
    """Write the result JSON and print the house one-line-per-predicate summary."""
    payload = dict(payload)
    payload['predictions'] = {k: bool(v) for k, v in preds.items()}
    payload['runtime_s'] = round(time.time() - t0, 1)
    payload['lib_version'] = LIB_VERSION
    _write_json(payload, out_path)
    print('\n' + ' | '.join(f'{k.split("_")[0]}_{k.split("_")[1]} {bool(v)}'
                            for k, v in preds.items()), flush=True)
    print(f'wrote {out_path} ({time.time() - t0:.1f}s)', flush=True)
