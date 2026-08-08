"""Byte-level BPE tokenizers trained on OUR corpus, and the tokenizer comparison.

WHY THIS EXISTS (Logan's catch, 2026-08-08).  The first corpus build reduced the
vocabulary by TRUNCATING GPT-2's 50,257 ids to the top-K and mapping the rest to
<UNK>.  That is a crude hack: it throws away 20.0% of tokens at V=4096 and 13.0%
at V=8192, makes UNK the most frequent symbol in every table, and buys no
compression at all (the segmentation is still GPT-2's, so a 512-token sequence
covers exactly the same text -- you just cannot read 13-20% of it).  A byte-level
BPE trained on THIS text has

  * ZERO UNK by construction -- the initial alphabet is all 256 bytes, so every
    possible input has a segmentation (byte fallback), and
  * better compression at the same V, because every merge is chosen for this data.

Small vocabularies are also not a compromise at our scale: "Scaling Laws with
Vocabulary: Larger Models Deserve Larger Vocabularies" (arXiv 2407.13623,
NeurIPS 2024) finds the compute-optimal vocabulary grows with the NON-vocabulary
parameter count, and our bodies are 25k-1.6M parameters.

TEXT SOURCE -- faithful and cheap.  The parent program committed GPT-2 *token id*
shards (../qk_mdl/corpus_fresh/shard00..06.npy, FineWeb sample-10BT docs
45367..267574).  We recover the text by DECODING those ids with the GPT-2
tokenizer rather than re-streaming FineWeb.  This is exactly faithful -- GPT-2's
byte-level BPE is a lossless bijection on the byte string, verified here by a
round-trip control (ids -> text -> ids is bit-identical) -- and it is cheap
(~20 s for the whole 123M-token train split).  Re-streaming FineWeb would risk a
different dataset revision re-sharding the doc indexing, which is exactly the
failure the parent's own builder warns about.  Documents are the segments between
GPT-2 <|endoftext|> (50256) ids; the first and last segment of a split region are
partial documents, which is inherent to the parent's fixed-length chunking.

THE BITS-PER-BYTE RULE (now program policy, see README.md).  Per-token
cross-entropy is NOT comparable across tokenizers: a tokenizer with fewer
bytes/token makes each prediction easier and inflates the token count, so nats
per token can move either way for reasons that have nothing to do with modelling
quality.  Every cross-tokenizer number in this program is reported in
BITS PER BYTE:
        bits/byte = (CE_nats_per_token * n_tokens) / (ln 2 * n_bytes)
with n_bytes the UTF-8 length of the SAME held text for every tokenizer.

Stages
    python tf_tokenizer.py train      # train BPE at 2048/4096/8192
    python tf_tokenizer.py compare    # the six-way measurement table
    python tf_tokenizer.py corpus 8192 4096   # rebuild the training corpus
    python tf_tokenizer.py controls   # round-trip / determinism / monotonicity
    python tf_tokenizer.py all
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/workspace/tensor_language/basis_aligned/qk_mdl/corpus_fresh'
GPT2_V = 50257
GPT2_EOS = 50256
SEQ = 513
SHARD_SEQS = 45000
N_TOTAL = 300000

# identical split boundaries to tf_corpus.py -- disjointness is inherited, not
# re-derived: the same SOURCE ROW RANGES define the same TEXT regions.
SPLITS = {'train': (0, 240000), 'held': (240000, 246000),
          'est': (246000, 276000), 'spare': (276000, 300000)}
TARGET_ROWS = {'train': 240000, 'held': 6000, 'est': 30000, 'spare': 24000}

BPE_VOCABS = (2048, 4096, 8192)
BPE_FIT_ROWS = 240000      # the WHOLE train split trains the merges (12 s), which
                           # matches tf_corpus.py's discipline exactly: vocabulary
                           # selection sees the train split and nothing else
STAT_FIT_ROWS = 40000      # train-split rows used to fit the comparison n-grams
ALPHAS = (1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


# ----------------------------------------------------------------- source text
_GPT2 = None


def gpt2():
    global _GPT2
    if _GPT2 is None:
        from transformers import AutoTokenizer
        import logging
        logging.getLogger('transformers').setLevel(logging.ERROR)
        _GPT2 = AutoTokenizer.from_pretrained('gpt2')
    return _GPT2


def source_rows(lo, hi):
    """Rows [lo:hi) of the parent program's concatenated GPT-2 shards."""
    rows, base = [], 0
    for i in range(7):
        p = f'{SRC}/shard{i:02d}.npy'
        if not os.path.exists(p):
            break
        a = np.load(p, mmap_mode='r')
        n = len(a)
        if base + n > lo and base < hi:
            rows.append(np.asarray(a[max(0, lo - base):min(n, hi - base)]))
        base += n
    assert rows, f'no source shards under {SRC} -- do NOT re-download'
    out = np.concatenate(rows)
    assert len(out) == hi - lo, f'source too small: {len(out)} < {hi - lo}'
    return out


def id_segments(arr):
    """Documents as GPT-2 id lists: the segments between <|endoftext|> ids."""
    ids = arr.reshape(-1).astype(np.int64)
    cut = np.flatnonzero(ids == GPT2_EOS)
    segs, prev = [], 0
    for c in cut:
        if c > prev:
            segs.append(ids[prev:c])
        prev = c + 1
    if prev < len(ids):
        segs.append(ids[prev:])
    return segs


def split_text(name, rows=None):
    """Decoded documents of one split region (list[str]).  `rows` caps the number
    of SOURCE rows used (for the BPE/statistics fits, which see train only)."""
    lo, hi = SPLITS[name]
    if rows is not None:
        hi = min(hi, lo + rows)
    segs = id_segments(source_rows(lo, hi))
    return gpt2().batch_decode([s.tolist() for s in segs])


def utf8_bytes(texts):
    return sum(len(t.encode('utf-8')) for t in texts)


# ------------------------------------------------------------- BPE training
def bpe_path(V):
    return f'{HERE}/tf_bpe_{V}.json'


def train_bpe(V, texts, save=True, show=True):
    """Byte-level BPE with byte fallback.  DETERMINISTIC: the HuggingFace BPE
    trainer has no stochastic component, so the merge list is a function of the
    training text and the settings alone (controlled in `controls`).

    `initial_alphabet = ByteLevel.alphabet()` is REQUIRED, not cosmetic: it forces
    all 256 byte symbols into the vocabulary whether or not they occur in the
    training text, which is what makes the UNK rate exactly 0 on ANY input."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers import processors
    tok = Tokenizer(models.BPE())                       # unk_token=None
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tr = trainers.BpeTrainer(
        vocab_size=V, min_frequency=2, show_progress=show,
        special_tokens=['<|endoftext|>'],               # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=tr, length=len(texts))
    got = tok.get_vocab_size()
    assert got == V, f'trained vocab {got} != {V}'
    assert tok.token_to_id('<|endoftext|>') == 0, 'EOS must be id 0'
    if save:
        tok.save(bpe_path(V))
    return tok


def load_bpe(V):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(bpe_path(V))


def encode_docs(tok, texts, eos=0, batch=20000):
    """One flat uint16-safe id stream: each document followed by EOS."""
    out = []
    for a in range(0, len(texts), batch):
        for e in tok.encode_batch(texts[a:a + batch], add_special_tokens=False):
            out.append(np.asarray(e.ids, dtype=np.int32))
            out.append(np.asarray([eos], dtype=np.int32))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


def gpt2_stream(texts, eos=GPT2_EOS, batch=20000):
    tk = gpt2()
    out = []
    for a in range(0, len(texts), batch):
        for ids in tk(texts[a:a + batch], add_special_tokens=False)['input_ids']:
            out.append(np.asarray(ids, dtype=np.int32))
            out.append(np.asarray([eos], dtype=np.int32))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


# ------------------------------------------------------- n-gram in bits/byte
def ngram_bits_per_byte(fit_ids, held_ids, V, n_bytes, alphas=ALPHAS):
    """Unigram floor and add-alpha bigram (unigram backoff), fitted on `fit_ids`
    and scored on `held_ids`, reported per token AND per byte.  Sparse bigram
    counting so V=50257 costs the same as V=2048."""
    fit_ids = fit_ids.astype(np.int64)
    held_ids = held_ids.astype(np.int64)
    uni = np.bincount(fit_ids, minlength=V).astype(np.float64)
    p_uni = (uni + 1.0) / (uni.sum() + V)
    ce_uni = float(-np.log(p_uni[held_ids[1:]]).mean())

    prev, nxt = fit_ids[:-1], fit_ids[1:]
    uk, uc = np.unique(prev * V + nxt, return_counts=True)
    rowsum = np.bincount(prev, minlength=V).astype(np.float64)

    def bigram_ce(data, alpha):
        p_, n_ = data[:-1], data[1:]
        k = p_ * V + n_
        idx = np.clip(np.searchsorted(uk, k), 0, len(uk) - 1)
        c = np.where(uk[idx] == k, uc[idx], 0).astype(np.float64)
        return float(-np.log((c + alpha * p_uni[n_])
                             / (rowsum[p_] + alpha)).mean())

    # alpha tuned on a held-out HALF of the held stream, scored on the other half
    # (the comparison table has no estimation split of its own)
    mid = len(held_ids) // 2
    sweep = {str(a): bigram_ce(held_ids[:mid], a) for a in alphas}
    best = float(min(sweep, key=lambda k: sweep[k]))
    ce_bi = bigram_ce(held_ids[mid:], best)
    ntok = len(held_ids) - 1
    f = ntok / (math.log(2) * n_bytes)
    return {'unigram_ce_nats_per_token': round(ce_uni, 5),
            'bigram_ce_nats_per_token': round(ce_bi, 5),
            'unigram_bits_per_byte': round(ce_uni * f, 5),
            'bigram_bits_per_byte': round(ce_bi * f, 5),
            'bigram_alpha': best,
            'bigram_alpha_at_grid_edge': best in (alphas[0], alphas[-1]),
            'bigram_beats_unigram': bool(ce_bi < ce_uni)}


def trunc_lut(V):
    """The truncation LUT tf_corpus.py built (train-split frequency order)."""
    p = f'{HERE}/tf_corpus_v{V}/lut_gpt2_to_new.npy'
    assert os.path.exists(p), f'{p} missing -- run tf_corpus.py first'
    return np.load(p).astype(np.int64)


def unk_repair_bits(gpt2_held, lut, gpt2_fit, n_bytes):
    """Bits per byte a truncated vocabulary must ADD to actually reproduce the
    text.  Mapping a token to UNK is LOSSY, so its bits/byte is not a code length
    for the text at all; the honest two-part code pays, for every UNK, the cost of
    naming which discarded GPT-2 id it was, under the train-split unigram
    restricted to the discarded set.  This is a LOWER bound on the repair cost
    (a real decoder would also need the identity of UNK-vs-kept, already paid)."""
    cnt = np.bincount(gpt2_fit.astype(np.int64), minlength=GPT2_V).astype(np.float64)
    discarded = (lut == 0)
    cnt = np.where(discarded, cnt + 1.0, 0.0)
    p = cnt / cnt.sum()
    h = gpt2_held.astype(np.int64)
    unk = h[lut[h] == 0]
    if len(unk) == 0:
        return 0.0
    return float(-np.log2(p[unk]).sum() / n_bytes)


# ------------------------------------------------------------ the comparison
def compare(out_json=None):
    t0 = time.time()
    out_json = out_json or f'{HERE}/tf_tokenizer_compare.json'
    print('decoding held text ...', flush=True)
    held_txt = split_text('held')
    n_bytes = utf8_bytes(held_txt)
    print(f'  held: {len(held_txt)} docs, {n_bytes} UTF-8 bytes', flush=True)
    print(f'decoding statistics-fit text (train rows 0:{STAT_FIT_ROWS}) ...',
          flush=True)
    fit_txt = split_text('train', rows=STAT_FIT_ROWS)
    print(f'  fit: {len(fit_txt)} docs, {utf8_bytes(fit_txt)} bytes '
          f'({time.time() - t0:.0f}s)', flush=True)

    g_held = gpt2_stream(held_txt)
    g_fit = gpt2_stream(fit_txt)
    rows = {}

    def record(name, V, held_ids, fit_ids, unk_id, extra=None):
        unk = (0.0 if unk_id is None
               else float((held_ids == unk_id).mean()))
        ng = ngram_bits_per_byte(fit_ids, held_ids, V, n_bytes)
        r = {'vocab_size': V, 'n_tokens_held': int(len(held_ids)),
             'n_bytes_held': int(n_bytes),
             'bytes_per_token': round(n_bytes / len(held_ids), 4),
             'bytes_seen_by_a_512_token_sequence':
                 round(512 * n_bytes / len(held_ids), 1),
             'unk_rate': round(unk, 6), 'lossy': bool(unk > 0)}
        r.update(ng)
        r.update(extra or {})
        rows[name] = r
        print(f'  {name:24s} b/tok {r["bytes_per_token"]:6.3f}  UNK '
              f'{r["unk_rate"]*100:5.2f}%  tokens {r["n_tokens_held"]:9d}  '
              f'uni {r["unigram_bits_per_byte"]:.4f} bpb  bi '
              f'{r["bigram_bits_per_byte"]:.4f} bpb', flush=True)

    print('measuring ...', flush=True)
    record('gpt2-50257', GPT2_V, g_held, g_fit, None)
    for V in (4096, 8192):
        lut = trunc_lut(V)
        rep = unk_repair_bits(g_held, lut, g_fit, n_bytes)
        record(f'truncGPT2-{V}', V, lut[g_held], lut[g_fit], 0,
               extra={'unk_repair_bits_per_byte': round(rep, 5),
                      'note': 'LOSSY code: UNK is unrecoverable, so its '
                              'bits/byte is NOT a code length for the text. '
                              'Add unk_repair_bits_per_byte for the honest '
                              'two-part cost.'})
        r = rows[f'truncGPT2-{V}']
        for k in ('unigram', 'bigram'):
            r[f'{k}_bits_per_byte_honest'] = round(
                r[f'{k}_bits_per_byte'] + rep, 5)
    for V in BPE_VOCABS:
        tok = load_bpe(V)
        record(f'newBPE-{V}', V, encode_docs(tok, held_txt),
               encode_docs(tok, fit_txt), None,
               extra={'note': 'byte-level BPE trained on this corpus; UNK '
                              'impossible by construction (256-byte alphabet)'})

    res = {'held_bytes': int(n_bytes), 'held_docs': len(held_txt),
           'fit_rows': STAT_FIT_ROWS, 'bpe_fit_rows': BPE_FIT_ROWS,
           'rule': 'per-token CE is NOT comparable across tokenizers; every '
                   'cross-tokenizer number here is bits/byte on the SAME held '
                   'text (same n_bytes denominator for every row)',
           'rows': rows, 'seconds': round(time.time() - t0, 1)}
    # control: bytes/token monotone in vocabulary size for the trained BPEs
    bt = [rows[f'newBPE-{V}']['bytes_per_token'] for V in BPE_VOCABS]
    res['control_bpe_bytes_per_token_monotone'] = bool(
        all(x < y for x, y in zip(bt, bt[1:]))
        and bt[-1] < rows['gpt2-50257']['bytes_per_token'])
    res['control_bpe_zero_unk'] = bool(
        all(rows[f'newBPE-{V}']['unk_rate'] == 0.0 for V in BPE_VOCABS))
    # KNOWN-ANSWER control on the UNK-repair accounting.  By the chain rule, the
    # two-part code (unigram over {kept ids, UNK}) + (unigram over discarded ids
    # given UNK) is EXACTLY the full-vocabulary unigram code -- the split into
    # "which bucket" and "which member" is lossless.  So the honest unigram
    # bits/byte of every truncated vocabulary must equal gpt2-50257's to within
    # smoothing noise.  If it does not, the repair accounting is wrong.
    ref = rows['gpt2-50257']['unigram_bits_per_byte']
    dev = {f'truncGPT2-{V}': round(
        rows[f'truncGPT2-{V}']['unigram_bits_per_byte_honest'] - ref, 6)
        for V in (4096, 8192)}
    res['control_unk_repair_reproduces_full_unigram'] = {
        'pass': all(abs(v) < 1e-3 for v in dev.values()),
        'reference_bits_per_byte': ref, 'deviation': dev,
        'why': 'chain rule: bucket code + within-bucket code == full code'}
    json.dump(res, open(out_json, 'w'), indent=2)
    print(f'wrote {out_json} ({res["seconds"]}s)', flush=True)
    return res


# ------------------------------------------------------------- corpus rebuild
def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_corpus(V):
    """Rebuild train/held/est/spare with the trained BPE at vocabulary V.

    Same discipline as tf_corpus.py: the SAME source row ranges define the same
    disjoint TEXT regions, sequence length 513, shard sha256 in the manifest so
    the scale box reproduces byte-identically by running this script.  The BPE
    stream is ~34% longer than GPT-2's on the same text, so each region yields
    MORE than its target rows; we take the target-row PREFIX, which keeps the
    row counts (and therefore the single-epoch arithmetic and every downstream
    path) identical to the truncated build."""
    tok = load_bpe(V)
    out = f'{HERE}/tf_corpus_b{V}'
    os.makedirs(out, exist_ok=True)
    man = {'vocab_size': V, 'unk_id': -1, 'eos_id': 0, 'seq_len': SEQ,
           'tokenizer': f'tf_bpe_{V}.json (byte-level BPE trained on the train '
                        f'split text, rows 0:{BPE_FIT_ROWS})',
           'tokenizer_sha256': sha256(bpe_path(V)),
           'source': 'text decoded from basis_aligned/qk_mdl/corpus_fresh/'
                     'shard00..06.npy (FineWeb sample-10BT, docs 45367..267574); '
                     'GPT-2 decode is a lossless bijection, round-trip controlled',
           'shard_seqs': SHARD_SEQS,
           'splits': {k: list(v) for k, v in SPLITS.items()},
           'split_semantics': 'source ROW ranges define disjoint TEXT regions; '
                              'each region is re-tokenized independently and '
                              'truncated to its target row count',
           'target_rows': dict(TARGET_ROWS),
           'vocab_selected_on': 'train split text only (rows 0:%d)' % BPE_FIT_ROWS,
           'single_epoch': {'steps': 15000, 'batch': 16, 'seqs': 240000,
                            'predicted_tokens': 15000 * 16 * (SEQ - 1)},
           'splits_stats': {}, 'files': {}}
    for name in ('train', 'held', 'est', 'spare'):
        t0 = time.time()
        txt = split_text(name)
        nb = utf8_bytes(txt)
        ids = encode_docs(tok, txt)
        nrow = len(ids) // SEQ
        need = TARGET_ROWS[name]
        assert nrow >= need, (f'{name}: BPE stream yields {nrow} rows < target '
                              f'{need} -- region too small')
        mapped = ids[:need * SEQ].reshape(need, SEQ).astype(np.uint16)
        assert mapped.max() < V
        man['splits_stats'][name] = {
            'n_seqs': int(need), 'n_tokens': int(mapped.size),
            'unk_rate_per_token': 0.0, 'unk_rate_per_seq_mean': 0.0,
            'unk_rate_per_seq_p50': 0.0, 'unk_rate_per_seq_p95': 0.0,
            'unk_rate_per_seq_max': 0.0, 'frac_seqs_with_any_unk': 0.0,
            'region_bytes': int(nb), 'region_tokens': int(len(ids)),
            'region_rows_available': int(nrow),
            'region_bytes_per_token': round(nb / len(ids), 4),
            'frac_of_region_text_used': round(need * SEQ / len(ids), 4),
            'eos_rate_per_token': round(float((mapped == 0).mean()), 6)}
        files = []
        for s, a in enumerate(range(0, need, SHARD_SEQS)):
            p = f'{out}/{name}{s:02d}.npy'
            np.save(p, mapped[a:a + SHARD_SEQS])
            files.append({'file': os.path.basename(p),
                          'rows': int(min(SHARD_SEQS, need - a)),
                          'sha256': sha256(p)})
        man['files'][name] = files
        st = man['splits_stats'][name]
        print(f'   {name:6s} {need:7d} rows  ({nrow} available)  '
              f'b/tok {st["region_bytes_per_token"]:.3f}  '
              f'uses {st["frac_of_region_text_used"]*100:.1f}% of region text  '
              f'({time.time() - t0:.0f}s)', flush=True)
        del ids, mapped, txt

    vocab = {'vocab_size': V, 'unk_id': -1, 'eos_id': 0,
             'tokenizer_file': f'tf_bpe_{V}.json',
             'tokens': [None] * V, 'decoded': [None] * V}
    inv = {i: s for s, i in tok.get_vocab().items()}
    for i in range(V):
        piece = inv[i]
        vocab['tokens'][i] = piece
        try:
            vocab['decoded'][i] = tok.decoder.decode([piece])
        except Exception:                                   # noqa: BLE001
            vocab['decoded'][i] = piece
    json.dump(vocab, open(f'{HERE}/tf_vocab_b{V}.json', 'w'))
    man['vocab_file'] = f'tf_vocab_b{V}.json'
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=2)
    print(f'wrote {out}/MANIFEST.json', flush=True)
    return man


# ----------------------------------------------------------------- controls
def controls(out_json=None):
    """Positive controls, all hard.
      C1 GPT-2 decode round-trip: ids -> text -> ids is bit-identical, so the
         decoded text IS the corpus text (this licenses the whole approach).
      C2 BPE round-trip: encode -> decode is exact string equality on held text.
      C3 zero UNK: every held token id is a real vocabulary entry, and all 256
         byte symbols are present in the vocabulary (byte fallback).
      C4 determinism: retraining on the same text with the same settings gives
         a bit-identical merge list.
      C5 monotonicity: bytes/token strictly increases with vocabulary size.
      C6 the rebuilt corpus's bigram baseline beats its unigram floor (run by
         tf_train.py baselines; recorded here if the JSON exists)."""
    out_json = out_json or f'{HERE}/tf_tokenizer_controls.json'
    res, t0 = {}, time.time()

    # C1 -----------------------------------------------------------------
    arr = source_rows(0, 400)
    segs = id_segments(arr)
    txt = gpt2().batch_decode([s.tolist() for s in segs])
    re_ids = gpt2()(txt, add_special_tokens=False)['input_ids']
    ok = all(list(a) == list(b) for a, b in zip(re_ids, [s.tolist() for s in segs]))
    res['C1_gpt2_decode_roundtrip_exact'] = bool(ok)
    print(f'C1 gpt2 id->text->id exact on {len(segs)} docs: {ok}', flush=True)

    # C2/C3 ---------------------------------------------------------------
    held = split_text('held')[:4000]
    res['C2_bpe_roundtrip_exact'] = {}
    res['C3_zero_unk'] = {}
    for V in BPE_VOCABS:
        tok = load_bpe(V)
        enc = tok.encode_batch(held, add_special_tokens=False)
        dec = [tok.decode(e.ids) for e in enc]
        exact = all(a == b for a, b in zip(dec, held))
        bad = [i for i, (a, b) in enumerate(zip(dec, held)) if a != b][:3]
        res['C2_bpe_roundtrip_exact'][str(V)] = {
            'pass': bool(exact), 'n_docs': len(held), 'first_mismatches': bad}
        from tokenizers import pre_tokenizers
        vs = set(tok.get_vocab())
        alpha_ok = set(pre_tokenizers.ByteLevel.alphabet()) <= vs
        ids = np.concatenate([np.asarray(e.ids) for e in enc])
        res['C3_zero_unk'][str(V)] = {
            'pass': bool(alpha_ok and ids.max() < V and ids.min() >= 0),
            'all_256_byte_symbols_in_vocab': bool(alpha_ok),
            'unk_token': tok.model.unk_token, 'unk_rate': 0.0}
        print(f'C2/C3 V={V}: roundtrip exact {exact}, 256-byte alphabet '
              f'{alpha_ok}', flush=True)

    # C4 ------------------------------------------------------------------
    small = split_text('train', rows=2000)
    a = train_bpe(1024, small, save=False, show=False)
    b = train_bpe(1024, small, save=False, show=False)
    ha = hashlib.sha256(json.dumps(json.loads(a.to_str())['model']['merges'],
                                   sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(json.loads(b.to_str())['model']['merges'],
                                   sort_keys=True).encode()).hexdigest()
    res['C4_merge_list_deterministic'] = {'pass': ha == hb, 'sha256': ha}
    print(f'C4 merge-list determinism: {ha == hb} ({ha[:16]})', flush=True)

    # C5 ------------------------------------------------------------------
    cp = f'{HERE}/tf_tokenizer_compare.json'
    if os.path.exists(cp):
        c = json.load(open(cp))
        res['C5_bytes_per_token_monotone'] = {
            'pass': c['control_bpe_bytes_per_token_monotone'],
            'bytes_per_token': {k: v['bytes_per_token']
                                for k, v in c['rows'].items()}}
        print('C5 bytes/token monotone: '
              f'{res["C5_bytes_per_token_monotone"]["pass"]}', flush=True)

    # C6 ------------------------------------------------------------------
    res['C6_bigram_beats_unigram'] = {}
    for V in BPE_VOCABS:
        bp = f'{HERE}/tf_baselines_b{V}.json'
        if os.path.exists(bp):
            d = json.load(open(bp))
            res['C6_bigram_beats_unigram'][str(V)] = {
                'pass': d['control_bigram_beats_unigram'],
                'unigram': d['unigram_floor_held_ce'],
                'bigram': d['bigram_held_ce']}
    res['seconds'] = round(time.time() - t0, 1)

    def allpass(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, dict):
            return all(allpass(v) for k, v in x.items()
                       if k == 'pass' or isinstance(v, dict))
        return True
    res['ALL_PASS'] = bool(allpass(res))
    json.dump(res, open(out_json, 'w'), indent=2)
    print(f'ALL_PASS = {res["ALL_PASS"]}  -> {out_json}', flush=True)
    return res


# --------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    args = [int(a) for a in sys.argv[2:] if a.isdigit()]
    t0 = time.time()
    if cmd in ('train', 'all'):
        print(f'decoding BPE training text (train rows 0:{BPE_FIT_ROWS}) ...',
              flush=True)
        txt = split_text('train', rows=BPE_FIT_ROWS)
        print(f'  {len(txt)} docs, {utf8_bytes(txt)/1e6:.1f} MB '
              f'({time.time() - t0:.0f}s)', flush=True)
        for V in (args or BPE_VOCABS):
            t1 = time.time()
            train_bpe(V, txt)
            print(f'  trained V={V} -> {bpe_path(V)} '
                  f'({time.time() - t1:.0f}s)', flush=True)
        del txt
    if cmd in ('compare', 'all'):
        compare()
    if cmd == 'corpus':
        for V in (args or [8192, 4096]):
            print(f'== building tf_corpus_b{V}', flush=True)
            build_corpus(V)
    if cmd == 'all':
        for V in (8192, 4096):
            print(f'== building tf_corpus_b{V}', flush=True)
            build_corpus(V)
    if cmd in ('controls', 'all'):
        controls()
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
