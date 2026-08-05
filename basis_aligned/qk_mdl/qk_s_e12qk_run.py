"""Wide-addressing arm (E12aqk): the second half of the narrowing-cost
mechanism question.

E12b showed narrow MESSAGES are recoverable: values shared from the wide
block recover -0.084 of E12a's +0.120 narrowing cost. This arm tests the
other channel -- ADDRESSING. E12aqk = E12a (264 wide -> 208 narrow, per-
block values from the narrow stream, NO value sharing) but each narrow
block's q/k/q2/k2 read the WIDE post-block-0 stream xn0 (full-bandwidth
addressing, per-block weights Dw->Dn). If addressing recovers cost like
values did, the narrowing bottleneck is symmetric; if not, it's messages.

Penalty: wide-input q/k matrices are grouped by the wide slot partition
(Gw=24); narrow-input reads (c_v, Left, Right) stay on Gn=26 -- handled by
the custom_group_penalty override (V8T.group_penalty dispatches to it).

argv[1] in {aqk, bqk}: bqk = same + shared values (composition arm, run
after aqk isolates the addressing term). Own JSON per arm, no races.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
import sys

import torch

import qk_tokenline_train as Q
Q.gpu_guard = lambda *a, **k: None

import qk_e_common as E
from qk_e_common import F

nn = torch.nn

_orig_oldheld = E.oldheld_record


def oldheld_flagged(stem, factory, jp, key):
    rec = _orig_oldheld(stem, factory, jp, key)
    if rec is not None:
        rec = dict(rec)
        rec['CAVEAT'] = ('scale box: old-held corpus is a SUBSTITUTE '
                         '(fresh34k[0:6000], pure eval) -- valid fresh eval, '
                         'NOT comparable to cooc old-held numbers')
        E.merge(jp, key, rec)
    return rec


E.oldheld_record = oldheld_flagged

M = importlib.import_module('qk_e12_funnel_run')
DEPTH = M.DEPTH

ARM = sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1 else 'aqk'
SHARED_V = (ARM == 'bqk')


class QKFunnel(M.FunnelRoute):
    """FunnelRoute with per-narrow-block q/k/q2/k2 reading the wide xn0."""

    def __init__(self, variant, cfg, shared_values=False):
        super().__init__(variant, cfg, shared_values=shared_values)
        for blk in self.nh:
            for nm in ('c_q', 'c_k', 'c_q2', 'c_k2'):
                setattr(blk, nm, nn.Linear(self.Dw, self.Dn, bias=False))

    def custom_group_penalty(self):
        tot = None

        def add(W, G):
            nonlocal tot
            S = W.shape[1] // G
            g = (W.pow(2).view(W.shape[0], G, S).sum(dim=(0, 2))
                 + 1e-12).sqrt().sum()
            tot = g if tot is None else tot + g

        for blk in [self.wb] + list(self.nh):
            for nm in E.READ_NAMES:
                lin = getattr(blk, nm, None)
                if lin is None:
                    continue
                add(lin.weight,
                    self.Gw if lin.weight.shape[1] == self.Dw else self.Gn)
        return tot

    def forward(self, idx, collect=None, nsub=None):
        B, Tq = idx.shape
        cw = self.cos_w[None, :Tq, None, :]
        sw = self.sin_w[None, :Tq, None, :]
        cn = self.cos_n[None, :Tq, None, :]
        sn = self.sin_n[None, :Tq, None, :]
        e = F.rms_norm(self.wte(idx), (self.Dw,))
        hn = self._snorm(e, self.Gw)
        v0 = self.wb.c_v(hn)
        y = self._attn(self.wb, hn, v0.view(B, Tq, self.NHw, self.HDw),
                       B, Tq, self.NHw, self.HDw, cw, sw)
        aw0 = self.wb.c_proj(y) * self.wmask_w[0].to(y.dtype) \
            if self.control else self.wb.c_proj(y)
        x0 = e + aw0
        xn0 = self._snorm(x0, self.Gw)
        mw0 = self.wb.Down(self.wb.Left(xn0) * self.wb.Right(xn0)) \
            + self.wb.Down_bias
        if self.control:
            mw0 = mw0 * self.wmask_w[1].to(mw0.dtype)
        a0n = self.P_a(aw0) * self.srcmask[0].to(aw0.dtype)
        m0n = self.P_m(mw0) * self.srcmask[1].to(mw0.dtype)
        nsrc = [a0n, m0n]
        v_sh = None
        if self.shared_values:
            v_sh = self.P_sv(v0).view(B, Tq, self.NHn, self.HDn)
        if collect is not None:
            collect['wide_write'] = [aw0.detach(), mw0.detach()]
            if 'neck' in collect:
                collect['neck'] = (a0n + m0n).detach()

        def nentry(li):
            sub = nsub.get(li) if nsub is not None else None
            tot = e if self.control else None
            for si in range(self.n_visible(li)):
                s = sub[si] if (sub is not None and si in sub) else nsrc[si]
                tot = s if tot is None else tot + s
            return tot

        for lb, blk in enumerate(self.nh):
            li = lb + 1
            x = nentry(li)
            if collect is not None:
                collect.setdefault('entry_norm', []).append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'nentry' in collect:
                    collect['nentry'].append(x.detach())
            hnn = self._snorm(x, self.Gn)
            v = v_sh if self.shared_values else \
                blk.c_v(hnn).view(B, Tq, self.NHn, self.HDn)

            def qkw(lin):
                z = lin(xn0).view(B, Tq, self.NHn, self.HDn)
                return Q.apply_rot(F.rms_norm(z, (self.HDn,)), cn, sn)

            q, k = qkw(blk.c_q), qkw(blk.c_k)
            q2, k2 = qkw(blk.c_q2), qkw(blk.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / self.HDn
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / self.HDn
            pat = (s1 * s2).masked_fill(~self.mask[:Tq, :Tq], 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, self.Dn)
            aw = blk.c_proj(y) * self.wmask_n[2 * lb].to(y.dtype)
            x = x + aw
            xnn = self._snorm(x, self.Gn)
            mw = blk.Down(blk.Left(xnn) * blk.Right(xnn)) + blk.Down_bias
            mw = mw * self.wmask_n[2 * lb + 1].to(mw.dtype)
            if collect is not None:
                collect.setdefault('nattn', []).append(aw.detach())
                collect.setdefault('nmlp', []).append(mw.detach())
            nsrc.append(aw)
            nsrc.append(mw)
        x = nentry(DEPTH)
        x = F.rms_norm(x, (self.Dn,))
        logits = self.W_up(x) @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_qk():
    torch.manual_seed(Q.SEED)
    return QKFunnel(f'E12{ARM}', M.cfg_a(),
                    shared_values=SHARED_V).to(E.DEV)


if __name__ == '__main__':
    M.JP = E.jpath(f'qk_e12_{ARM}.json')
    E.setup()
    with torch.no_grad():
        m = make_qk().eval().float()
        out = m(Q.HELD[:2, :Q.T])
        assert torch.isfinite(out).all()
        p = float(m.custom_group_penalty())
        body = sum(q.numel() for n, q in m.named_parameters() if 'wte' not in n)
        print(f"smoke: finite out, penalty {p:.1f}, body params {body}",
              flush=True)
        del m
        torch.cuda.empty_cache()
    M.run_arm(f'qk_e12_{ARM}', f'E12{ARM}', make_qk,
              ('wide addressing: narrow blocks q/k/q2/k2 read the wide xn0 '
               '(Dw->Dn per block), values '
               + ('SHARED from wide c_v (composition arm)' if SHARED_V else
                  'per-block from the narrow stream (isolates addressing)')),
              extra_pairs=(('qk_e12_a', 'e12a'), ('qk_e12_b', 'e12b')))
    print(f'e12 {ARM} done', flush=True)
