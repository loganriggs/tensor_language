"""Runnable skeleton of the architecture-agnostic replacement harness
(HARNESS.md). Three parts: the TracedModel interface an adapter implements per
architecture (~100 lines, the only model-specific code), the generic
ReplacementHarness (fit / sequential refit / joint scoring in both norm regimes
/ greedy allocation / mandatory self-tests), and a Bilin18Traced adapter that
wraps the existing bilin18_pipe_refit machinery so the skeleton runs on bilin18.
__main__ runs the self-test battery only (a few short forwards; no experiments).
"""
import sys, time, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F

# ---------------------------------------------------------------- generic ----

class TracedModel:
    """Per-architecture adapter. THE contract (HARNESS.md section 1): capture and
    apply happen inside the SAME forward implementation; no hooks on modules a
    manual forward never calls. Assignment format everywhere:
    {component: {'kind':'full'|'const'|'linear','W','bx','by','rank'}}."""
    n_layers = None; d_model = None
    def components(self): raise NotImplementedError   # ordered front-to-back
    def forward_with(self, idx, assignment, capture=None, gain_frozen=False):
        """Return (logits_float, cap). cap = (X, Y) flattened float (input,
        TRUE component output) of `capture` under the installed hybrid."""
        raise NotImplementedError
    def reference_ce(self, idx): raise NotImplementedError          # self-test 2
    def reference_mean_ablate_ce(self, idx, comp, mean):            # self-test 3
        raise NotImplementedError
    def reference_fit(self, comp, assignment): return None          # self-test 5
    def component_params(self, comp): raise NotImplementedError     # balanced gauge
    def train_batches(self): raise NotImplementedError
    def eval_batches(self, quick=False): raise NotImplementedError


class ReplacementHarness:
    TOL_IDENT, TOL_XCHK, TOL_FROZEN = 1e-6, 2e-3, 1e-3

    def __init__(self, tm):
        self.tm = tm; self.map_cache = {}

    @torch.no_grad()
    def joint_ce(self, assignment, gain_frozen=False, quick=False):
        """JOINT held-out CE of the assignment (per-component sums are not a
        score -- HARNESS.md section 4). Both norm regimes via gain_frozen."""
        tot, n = 0.0, 0
        for b in self.tm.eval_batches(quick):
            lg, _ = self.tm.forward_with(b[:, :-1].contiguous(), assignment,
                                         gain_frozen=gain_frozen)
            ce = F.cross_entropy(lg.view(-1, lg.size(-1)), b[:, 1:].reshape(-1))
            tot += float(ce) * b[:, 1:].numel(); n += b[:, 1:].numel()
        return tot / n

    @torch.no_grad()
    def capture(self, comp, assignment):
        xs, ys = [], []
        for b in self.tm.train_batches():
            _, cap = self.tm.forward_with(b, assignment, capture=comp)
            xs.append(cap[0]); ys.append(cap[1])
        return torch.cat(xs), torch.cat(ys)

    @torch.no_grad()
    def fit_layer(self, comp, assignment, rank=None):
        """Ridge fit of comp's (input -> true output) captured UNDER
        `assignment` (so a front-to-back sweep is a sequential refit), then
        SVD-truncated to `rank`. rank 0 -> constant mean; None -> full."""
        X, Y = self.capture(comp, assignment)
        d = X.shape[1]; bx = X.mean(0); by = Y.mean(0)
        if rank == 0:
            return {'kind': 'const', 'W': torch.zeros(d, d, device=X.device),
                    'bx': bx, 'by': by, 'rank': 0}
        Xc, Yc = X - bx, Y - by
        lam = 1e-2 * float((Xc**2).mean()) * d / Xc.shape[0]
        W = torch.linalg.solve(Xc.T @ Xc / Xc.shape[0]
                               + lam * torch.eye(d, device=X.device),
                               Xc.T @ Yc / Xc.shape[0])
        mp = {'kind': 'linear', 'W': W, 'bx': bx, 'by': by, 'rank': d}
        return mp if rank is None or rank >= d else self._truncate(mp, rank)

    @staticmethod
    def _truncate(mp, r):
        if r == 0:
            return {'kind': 'const', 'W': torch.zeros_like(mp['W']),
                    'bx': mp['bx'], 'by': mp['by'], 'rank': 0}
        U, S, Vh = torch.linalg.svd(mp['W'])
        return {'kind': 'linear', 'W': U[:, :r] @ torch.diag(S[:r]) @ Vh[:r],
                'bx': mp['bx'], 'by': mp['by'], 'rank': r}

    def refit_sweep(self, plan):
        """plan: ordered [(comp, rank)] front-to-back. Each stand-in is fit on
        the model with all upstream stand-ins INSTALLED (the sequential-refit
        default, section 158's frontier lever)."""
        assignment = {}
        for comp, rank in plan:
            assignment[comp] = self.fit_layer(comp, dict(assignment), rank)
        return assignment

    def _cached_map(self, comp, assignment):
        """Rank-64 refit map cache. Key includes the UPSTREAM component set, so
        any upstream change invalidates (stale map -> automatic refit)."""
        order = {c: i for i, c in enumerate(self.tm.components())}
        key = (comp, tuple(sorted(c for c in assignment
                                  if order[c] < order[comp])))
        if key not in self.map_cache:
            self.map_cache[key] = self.fit_layer(comp, assignment, 64)
        return self.map_cache[key]

    def standin_params(self, st):
        d = self.tm.d_model
        return 0 if st['kind'] == 'full' else d if st['kind'] == 'const' \
            else 2 * d * st['rank']

    def param_count(self, assignment, standins_only=True):
        tot = sum(self.standin_params(st) for st in assignment.values())
        if not standins_only:
            tot += sum(self.tm.component_params(c) for c in self.tm.components()
                       if assignment.get(c, {'kind': 'full'})['kind'] == 'full')
        return tot

    def greedy_allocate(self, budget_nats, ranks=(0, 4, 16)):
        """Greedy assignment under a joint-CE budget. Marginals are APPROXIMATE
        (cached rank-64 map truncations, quick eval); acceptance is decided by
        an EXACT joint rescore on the full eval set."""
        assignment, remaining = {}, list(self.tm.components())
        base = self.joint_ce({})
        while remaining:
            cands = []
            for comp in remaining:
                mp = self._cached_map(comp, assignment)
                for r in ranks:
                    st = self._truncate(mp, r)
                    trial = dict(assignment); trial[comp] = st
                    approx = self.joint_ce(trial, quick=True) - base
                    saved = self.tm.component_params(comp) - self.standin_params(st)
                    cands.append((approx / max(saved, 1), comp, st))
            cands.sort(key=lambda t: t[0])
            _, comp, st = cands[0]
            trial = dict(assignment); trial[comp] = st
            exact = self.joint_ce(trial) - base          # exact joint rescore
            if exact <= budget_nats:
                assignment = trial
            remaining.remove(comp)   # accepted, or unreplaceable at these ranks
        return assignment, self.joint_ce(assignment) - base if assignment else 0.0

    # ------------------------------------------------- mandatory self-tests --
    @torch.no_grad()
    def self_tests(self, comp=None):
        """Auto-generated gate (HARNESS.md section 2). Must be green before any
        experiment on this adapter. Fast: quick eval subset only."""
        tm = self.tm
        comp = comp or tm.components()[len(tm.components()) // 2]
        res = {}
        base = self.joint_ce({}, quick=True)

        ident = self.joint_ce({comp: {'kind': 'full'}}, quick=True)
        res['identity'] = abs(ident - base) <= self.TOL_IDENT

        ref = sum(tm.reference_ce(b) for b in tm.eval_batches(quick=True)) \
            / len(list(tm.eval_batches(quick=True)))
        res['noop_vs_reference'] = abs(base - ref) <= self.TOL_XCHK

        _, Y = self.capture(comp, {})
        ybar = Y.mean(0)
        st = {'kind': 'const', 'W': torch.zeros(tm.d_model, tm.d_model,
                                                device=Y.device),
              'bx': torch.zeros(tm.d_model, device=Y.device), 'by': ybar,
              'rank': 0}
        ours = self.joint_ce({comp: st}, quick=True)
        refm = sum(tm.reference_mean_ablate_ce(b, comp, ybar)
                   for b in tm.eval_batches(quick=True)) \
            / len(list(tm.eval_batches(quick=True)))
        res['mean_ablation_xcheck'] = abs(ours - refm) <= self.TOL_XCHK

        frozen = self.joint_ce({}, gain_frozen=True, quick=True)
        res['gain_freeze_noop'] = abs(frozen - base) <= self.TOL_FROZEN

        refit = tm.reference_fit(comp, {})
        if refit is not None:
            mine = self.fit_layer(comp, {})
            res['fit_xcheck'] = float((mine['W'] - refit['W']).abs().max()) <= 1e-4

        for k, v in res.items():
            print(f'  {k:22s} {"PASS" if v else "FAIL"}')
        print(f'self_tests: {"ALL GREEN" if all(res.values()) else "FAILED"} '
              f'(base {base:.4f})')
        return all(res.values()), res


# ------------------------------------------------------- bilin18 adapter ----

import bilin18_pipe_refit as PR
from bilin18_joint_removal import fwd as JR_fwd, PATCH as JR_PATCH, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH, HD, D = 9, 128, 1152


class Bilin18Traced(TracedModel):
    """Wraps bilin18_pipe_refit: PR.fwd_lin is the traced forward (free regime
    + capture), PR.fit_layer the reference fit. Components: the 18 MLPs (the
    machinery the pipe supports; attention stand-ins need forward extension)."""
    n_layers, d_model = 18, D

    def components(self): return [f'mlp{i}' for i in range(18)]

    def _lins(self, assignment):
        lins = {}
        for comp, st in assignment.items():
            if st['kind'] == 'full': continue
            lins[int(comp[3:])] = {'W': st['W'], 'bx': st['bx'], 'by': st['by']}
        return lins

    @torch.no_grad()
    def forward_with(self, idx, assignment, capture=None, gain_frozen=False):
        lins = self._lins(assignment)
        if gain_frozen:
            assert capture is None
            return self._fwd_frozen(idx, lins), None
        old = PR.LINS
        try:
            PR.LINS = lins
            return PR.fwd_lin(idx, want=int(capture[3:]) if capture else None)
        finally:
            PR.LINS = old

    @torch.no_grad()
    def _fwd_frozen(self, idx, lins):
        """Gain-frozen regime: clean + hybrid in lockstep, hybrid's final norm
        clamped to the clean run's per-token rms (section 116 instrument).
        Validated by self-tests: frozen == free at zero damage."""
        B, T = idx.shape
        cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        def block(li, x, x0, v1, hybrid):
            blk = m.transformer.h[li]; a = blk.attn; mlp = blk.mlp
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0; xin = x
            hcur = F.rms_norm(x, (D,))
            def qk(l):
                z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            v1n = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1n.view_as(v)
            q, k1, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v)
                             .reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,))
            if hybrid and li in lins:
                mp = lins[li]; xi = xin.reshape(-1, D).float()
                mo = ((xi - mp['bx']) @ mp['W'] + mp['by']).to(x.dtype).view_as(x)
            else:
                mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
            return x + mo, v1n
        xc = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = xc; xh = xc
        v1c = v1h = None
        for li in range(18):
            xc, v1c = block(li, xc, x0, v1c, False)
            xh, v1h = block(li, xh, x0, v1h, True)
        rms_c = xc.float().pow(2).mean(-1, keepdim=True).sqrt()
        xn = (xh.float() / rms_c.clamp_min(1e-8)).to(xh.dtype)
        return (30 * torch.tanh(m.lm_head(xn) / 30)).float()

    def reference_ce(self, idx):
        assert not JR_PATCH
        return float(JR_fwd(idx).mean())      # independent forward (joint_removal)

    def reference_mean_ablate_ce(self, idx, comp, mean):
        JR_PATCH[int(comp[3:])] = (torch.eye(D, device=DEV), mean)
        try: return float(JR_fwd(idx).mean())  # PATCH path, Q=I: a second impl
        finally: JR_PATCH.clear()

    def reference_fit(self, comp, assignment):
        old = PR.LINS
        try:
            PR.LINS = self._lins(assignment)
            return PR.fit_layer(int(comp[3:]))
        finally:
            PR.LINS = old

    def component_params(self, comp):
        """Live CP units counted at the balanced gauge point (WP1, closed form:
        each unit's three norms -> their geometric mean; dead units zeroed)."""
        mlp = m.transformer.h[int(comp[3:])].mlp
        na = mlp.Left.weight.float().norm(dim=1)
        nb = mlp.Right.weight.float().norm(dim=1)
        nc = mlp.Down.weight.float().norm(dim=0)
        mi = (na * nb * nc).pow(1.0 / 3.0)
        return int((mi > 1e-6 * mi.max()).sum()) * 3 * D

    def train_batches(self):
        return [FW[i:i + 6, :256].to(DEV) for i in range(0, 48, 6)]

    def eval_batches(self, quick=False):
        hi = 316 if quick else 380
        return [FW[i:i + 4, :257].to(DEV) for i in range(300, hi, 4)]


if __name__ == '__main__':
    t0 = time.time()
    h = ReplacementHarness(Bilin18Traced())
    ok, _ = h.self_tests(comp='mlp9')
    print(f'({time.time()-t0:.0f}s)')
    sys.exit(0 if ok else 1)
