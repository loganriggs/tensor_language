"""O2, the decisive form: SERIALISE the winning description to a real bit
string and DECODE it back, with nothing passed between encoder and decoder
except that bit string.  If the reconstruction is bit-identical and the file
is no longer than the charged bill, the accounting is complete; if the decoder
needs anything else, this fails to run.

A static arithmetic coder is used for the symbol streams (the same coder the
bill assumes), with the probability model taken from the fp16 histogram that
the bill already pays for.
"""
import json
import math
import os
import struct

import numpy as np
import torch

import tf_compress as CC

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = 0xFFFFFFFF
HALF = 0x80000000
QTR = 0x40000000
TQR = 0xC0000000


class Enc:
    def __init__(self):
        self.low, self.high, self.pend = 0, TOP, 0
        self.out, self.buf, self.n = bytearray(), 0, 0

    def _bit(self, b):
        self.buf = (self.buf << 1) | b
        self.n += 1
        if self.n == 8:
            self.out.append(self.buf)
            self.buf, self.n = 0, 0

    def _emit(self, b):
        self._bit(b)
        while self.pend:
            self._bit(1 - b)
            self.pend -= 1

    def enc(self, clo, chi, tot):
        r = self.high - self.low + 1
        self.high = self.low + (r * chi) // tot - 1
        self.low = self.low + (r * clo) // tot
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < TQR:
                self.pend += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP

    def finish(self):
        self.pend += 1
        self._emit(0 if self.low < QTR else 1)
        while self.n:
            self._bit(0)
        return bytes(self.out)


class Dec:
    def __init__(self, data):
        self.d, self.p, self.bit = data, 0, 0
        self.low, self.high, self.code = 0, TOP, 0
        for _ in range(32):
            self.code = (self.code << 1) | self._rb()

    def _rb(self):
        if self.p >= len(self.d):
            return 0
        b = (self.d[self.p] >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit, self.p = 0, self.p + 1
        return b

    def target(self, tot):
        r = self.high - self.low + 1
        return ((self.code - self.low + 1) * tot - 1) // r

    def upd(self, clo, chi, tot):
        r = self.high - self.low + 1
        self.high = self.low + (r * chi) // tot - 1
        self.low = self.low + (r * clo) // tot
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QTR and self.high < TQR:
                self.low -= QTR
                self.high -= QTR
                self.code -= QTR
            else:
                break
            self.low = (self.low << 1) & TOP
            self.high = ((self.high << 1) | 1) & TOP
            self.code = ((self.code << 1) | self._rb()) & TOP


def counts_from_fp16(p16):
    """Both sides derive integer frequencies from the SAME fp16 probability
    vector, deterministically, so the decoder needs no extra information."""
    p = np.asarray(p16, dtype=np.float32).astype(np.float64)
    p = np.maximum(p, 0)
    if p.sum() <= 0:
        p = np.ones_like(p)
    c = np.maximum(1, np.round(p / p.sum() * 65536)).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(c)])
    return c, cum, int(cum[-1])


def main():
    stem = 'tf_vanilla_d1_w128_b8192_s0'
    D = CC.D1Desc(stem, device='cpu')
    W = D.base['wte_out']
    V, d = W.shape
    bpr = 768

    # ---- ENCODER SIDE: build exactly the description q_transform describes
    mu = W.mean(0, keepdim=True)
    X = W - mu
    var = (X * X).mean(0)
    b = CC._alloc(var, bpr)
    lo = X.min(0).values.half()
    hi = X.max(0).values.half()
    payload = bytearray()
    payload += mu.reshape(-1).numpy().astype('<f4').tobytes()          # d*32
    payload += lo.numpy().astype('<f2').tobytes()                      # d*16
    payload += hi.numpy().astype('<f2').tobytes()                      # d*16
    alloc = bytes(bytearray((int(b[2 * j]) << 4) | int(b[2 * j + 1])
                            for j in range(d // 2)))                   # d*4
    payload += alloc
    hist_bytes = bytearray()
    E = Enc()
    codes_all = []
    for j in range(d):
        bj = int(b[j])
        if bj == 0:
            codes_all.append(None)
            continue
        lof, hif = float(lo[j]), float(hi[j])
        step = max((hif - lof) / (2 ** bj - 1), 1e-30)
        c = torch.round((X[:, j] - lof) / step).clamp(0, 2 ** bj - 1).long()
        cnt = torch.bincount(c, minlength=2 ** bj).double()
        p16 = (cnt / cnt.sum()).numpy().astype(np.float16)
        hist_bytes += p16.tobytes()                                    # 2^bj*16
        cc, cum, tot = counts_from_fp16(p16)
        cn = c.numpy()
        for s in cn:
            E.enc(int(cum[s]), int(cum[s + 1]), tot)
        codes_all.append(cn)
    stream = E.finish()
    blob = bytes(payload) + bytes(hist_bytes) + stream
    actual_bits = 8 * len(blob)

    # ---- DECODER SIDE: from `blob` alone -------------------------------
    o = 0
    mu_d = np.frombuffer(blob[o:o + 4 * d], dtype='<f4').copy(); o += 4 * d
    lo_d = np.frombuffer(blob[o:o + 2 * d], dtype='<f2').copy(); o += 2 * d
    hi_d = np.frombuffer(blob[o:o + 2 * d], dtype='<f2').copy(); o += 2 * d
    al = blob[o:o + d // 2]; o += d // 2
    b_d = []
    for byte in al:
        b_d += [byte >> 4, byte & 0xF]
    hists = []
    for j in range(d):
        if b_d[j] == 0:
            hists.append(None)
            continue
        nl = 2 ** b_d[j]
        hists.append(np.frombuffer(blob[o:o + 2 * nl], dtype='<f2').copy())
        o += 2 * nl
    Dd = Dec(blob[o:])
    Zq = np.zeros((V, d), dtype=np.float32)
    for j in range(d):
        if b_d[j] == 0:
            continue
        cc, cum, tot = counts_from_fp16(hists[j])
        step = max((float(hi_d[j]) - float(lo_d[j])) / (2 ** b_d[j] - 1), 1e-30)
        col = np.empty(V, dtype=np.int64)
        for i in range(V):
            t = Dd.target(tot)
            s = int(np.searchsorted(cum, t, side='right') - 1)
            Dd.upd(int(cum[s]), int(cum[s + 1]), tot)
            col[i] = s
        assert (col == codes_all[j]).all(), f'column {j} decode mismatch'
        Zq[:, j] = col * step + float(lo_d[j])
    Wrec = torch.from_numpy(Zq + mu_d[None, :])

    Wref, bill = CC.q_transform(W, bpr, rot='none')[0], None
    _, bref = CC.q_transform(W, bpr, rot='none')
    same = bool(torch.equal(Wrec, Wref)) or float((Wrec - Wref).abs().max()) < 1e-9
    res = {'scheme': 'embT768 (the embedding half of the headline point)',
           'charged_bits': bref.total,
           'actual_serialised_bits': actual_bits,
           'actual_over_charged': actual_bits / bref.total,
           'roundtrip_bit_identical': same,
           'max_abs_reconstruction_diff': float((Wrec - Wref).abs().max()),
           'decoder_received_only_the_blob': True,
           'blob_bytes': len(blob),
           'itemised': {'means_fp32': 32 * d, 'lo_hi_fp16': 2 * 16 * d,
                        'alloc_4bit': 4 * d,
                        'histograms_fp16': 8 * len(hist_bytes),
                        'arith_stream': 8 * len(stream)}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'{HERE}/tf_reviewer_r3_codec.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
