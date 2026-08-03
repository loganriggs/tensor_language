"""E4 TYPED TOKEN SLOT (fresh single-epoch protocol).

In the V8 base, the normed token embedding is summed into EVERY consumer's
entry across the full 264-dim stream, and module writes then relay token
information onward (the writes are heavily token-determined). E4 makes the
token relay ARCHITECTURAL: slot 0 (dims 0..10) is reserved as a token line
that always contains the current token's code -- the embedding's slot-0 dims,
RMS-normalized within the slot -- and modules CANNOT write to it (their write
masks exclude slot 0) while every consumer including the readout can read it.
The embedding's other 253 dims never enter the stream, so ALL token
information flows through the 11-dim learned token code (the embedding is
learned, so training chooses what to put in those 11 dims).

Slot allocation: with slot 0 reserved, 24 module writes share the remaining
23 slots: module k -> slot k+1 for k = 0..22, and block 11's MLP write shares
slot 23 with block 11's attention write (both only feed the readout; noted as
the one compromise -- 264 is not divisible into 25 equal slots).

Questions: (a) CE vs the fresh controls (paired vs E0a/E0b); (b) does the
token-determined variance of the OTHER slots' writes drop vs E0b (standard
token-determined probe, qk_v8_probe machinery) -- i.e. does an architectural
token relay free learned capacity?

Positive controls: (i) E4 with the token line DISABLED (tokline=False: full
normed embedding + standard slot map) reproduces the plain V8Route forward at
init to < 1e-4 -- certifies the overridden forward; (ii) with the token line
ON, every consumer entry's slot-0 dims equal the normed token code exactly and
no module write touches slot 0. Results -> qk_e4.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, V8T, C, W, DEPTH, F, torch

JP = E.jpath('qk_e4.json')
STEM = 'qk_e4_tokenslot'


def slot_of(k):
    """Module k's write slot under the E4 allocation."""
    return min(k + 1, 2 * DEPTH - 1)


class E4Route(V8T.V8Route):
    """V8Route with an architectural token line in slot 0."""

    def __init__(self, variant, depth, tokline=True):
        super().__init__(variant, depth)
        self.tokline = tokline
        if tokline:
            Dm = self.wte.weight.shape[1]
            S = Dm // (2 * depth)
            wm = torch.zeros(2 * depth, Dm)
            for k in range(2 * depth):
                s = slot_of(k)
                wm[k, S * s:S * (s + 1)] = 1.0
            with torch.no_grad():
                self.wmask.copy_(wm)

    def embed_stream(self, idx):
        e_raw = self.wte(idx)
        Dm = e_raw.shape[-1]
        if not self.tokline:
            return F.rms_norm(e_raw, (Dm,))
        S = Dm // (2 * self.depth)
        e = torch.zeros_like(e_raw)
        e[..., :S] = F.rms_norm(e_raw[..., :S], (S,))   # 11-dim token code
        return e

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = self.embed_stream(idx)
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = F.rms_norm(x, (Dm,))

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = F.rms_norm(x, (Dm,))
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e4(tokline=True):
    C.register('E4')
    torch.manual_seed(Q.SEED)
    return E4Route('E4', DEPTH, tokline=tokline).to(E.DEV)


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    # (i) token line off == plain V8Route (same seed, no extra RNG draws)
    base = C.make_variant('E4ctl').eval().float()
    moff = make_e4(tokline=False).eval().float()
    d = (moff(idx) - base(idx)).abs().max().item()
    print(f"control E4(tokline off)==V8Route at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-4
    del base, moff
    torch.cuda.empty_cache()
    # (ii) token line on: slot 0 of every entry IS the normed token code and
    # no write mask touches slot 0
    m = make_e4().eval().float()
    S = Q.D // (2 * DEPTH)
    assert float(m.wmask[:, :S].abs().sum()) == 0.0
    col = {'entry': [], 'entry_norm': [], 'attn_write': [], 'mlp_write': []}
    m(idx, collect=col)
    tok = F.rms_norm(m.wte(idx)[..., :S].float(), (S,))
    dmax = max(float((en.float()[..., :S] - tok).abs().max())
               for en in col['entry'])
    print(f"control E4 entries carry the token code in slot 0: max diff "
          f"{dmax:.2e} over {len(col['entry'])} entries", flush=True)
    assert dmax < 1e-4
    del m, col
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()
    m = make_e4()
    E.merge(JP, 'param_counts', W.param_counts(m))
    del m
    torch.cuda.empty_cache()
    E.train_arm(STEM, JP, 'E4', make_e4, E.GC)
    E.oldheld_record(STEM, make_e4, JP, 'E4_oldheld')
    E.paired_fresh(STEM, JP, 'E4')
    E.probe_arm(STEM, make_e4, JP, 'light_probe_E4', tok_key='tok_probe_E4',
                slot_of=slot_of)
    if not E.SMOKE:
        e0 = E.loadj(E.jpath('qk_e0.json'))
        e4 = E.loadj(JP)
        if 'tok_probe_E0b' in e0 and 'tok_probe_E4' in e4:
            E.merge(JP, 'token_determined_comparison', {
                'E0b_token_determined_mlp':
                    e0['tok_probe_E0b'].get('token_determined_mlp'),
                'E4_token_determined_mlp':
                    e4['tok_probe_E4'].get('token_determined_mlp'),
                'note': 'per-layer group-mean R^2 of the MLP writes; the E4 '
                        'question is whether these drop when the token relay '
                        'is architectural'})
    print('e4 tokenslot run done', flush=True)
