"""Reduced-vocabulary corpus for the tiny-full-interpretation program.

SOURCE: the SAME FineWeb sample-10BT stream the parent program (../qk_mdl) uses.
We do NOT re-download and we do NOT re-tokenize: the parent already committed
    ../qk_mdl/corpus_fresh/shard00..06.npy   (300,000 rows x 513 GPT-2 ids,
                                              docs 45367..267574, tokenizer gpt2)
so this builder is a pure deterministic REMAP of those shards.  That means the
scale box regenerates byte-identical data by running this script -- we only have
to commit the tiny vocabulary JSON, not ~300 MB of shards (the per-shard sha256
in the manifest is the byte-identity gate).

WHAT IT DOES
  1. counts GPT-2 token-id frequencies over the TRAIN split only (never the held
     or estimation split -- vocabulary selection must not see evaluation data),
  2. keeps the V-1 most frequent ids, assigns them new ids 1..V-1 in descending
     frequency order, and maps every other id to new id 0 = <UNK>,
  3. writes uint16 shards of the remapped rows plus a manifest with the UNK rate
     per token and per sequence for each split,
  4. saves the id mapping (new id -> GPT-2 id -> token string) so every V x V
     table this program materializes can be labelled with real token strings.

SPLITS (disjoint by construction; row indices into the concatenated 300,000)
    train  [     0 : 240000)   240,000 seqs   122,880,000 predicted tokens
    held   [240000 : 246000)     6,000 seqs   pure evaluation, never trained
    est    [246000 : 276000)    30,000 seqs   table fitting / behavioural stats
    spare  [276000 : 300000)    24,000 seqs   reserve (unused; keeps a margin)

SINGLE-EPOCH ARITHMETIC (why 240,000 train sequences)
  Every primary-grid cell trains ONE pass at batch 16 for TF_STEPS = 15,000 steps
  = 240,000 sequences = exactly the train split, each sequence visited once.
  The grid is 2 depths x 4 widths x 3 seeds = 24 models; every model reads the
  SAME 240,000 rows in the SAME order (epoch_order(0)), which is the parent
  program's convention -- freshness is a per-model property (no sequence is ever
  seen twice BY A MODEL), and sharing the order across cells is what makes the
  paired per-token comparisons between cells valid.  The largest cell (depth 2,
  width 256) therefore also sees fresh data for all 15,000 of its steps: the
  budget is data-limited, not step-limited, with 0 rows of slack by design and
  24,000 spare rows held in reserve if a later cell wants a longer schedule.

Usage:
    python tf_corpus.py             # builds V=4096 (primary) and V=8192 (check)
    python tf_corpus.py 4096        # one vocabulary only
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/workspace/tensor_language/basis_aligned/qk_mdl/corpus_fresh'
GPT2_V = 50257
SEQ = 513                      # 512 context tokens -> 512 targets, as in the parent
SHARD_SEQS = 45000             # ~46 MB/shard at uint16, the parent's convention
N_TOTAL = 300000

SPLITS = {'train': (0, 240000), 'held': (240000, 246000),
          'est': (246000, 276000), 'spare': (276000, 300000)}
TF_STEPS = 15000               # single-epoch step budget at batch 16
TF_BATCH = 16


def load_source():
    """Concatenate the parent program's GPT-2 shards in order (uint16)."""
    rows = []
    for i in range(7):
        p = f'{SRC}/shard{i:02d}.npy'
        if not os.path.exists(p):
            break
        rows.append(np.load(p, mmap_mode='r'))
    assert rows, f'no source shards under {SRC} -- do NOT re-download, fix the path'
    arr = np.concatenate([np.asarray(r) for r in rows])
    assert arr.shape[1] == SEQ, f'unexpected seq len {arr.shape[1]}'
    assert len(arr) == N_TOTAL, f'expected {N_TOTAL} source rows, got {len(arr)}'
    return arr


def count_ids(arr, lo, hi, chunk=20000):
    """GPT-2 id histogram over rows [lo:hi) (chunked; the split is ~123M ids)."""
    cnt = np.zeros(GPT2_V, dtype=np.int64)
    for a in range(lo, hi, chunk):
        b = min(a + chunk, hi)
        cnt += np.bincount(arr[a:b].reshape(-1).astype(np.int64),
                           minlength=GPT2_V)
    return cnt


def build_lut(counts, V):
    """new id 0 = UNK; ids 1..V-1 = the V-1 most frequent GPT-2 ids, descending.
    Ties broken by ascending GPT-2 id so the build is fully deterministic."""
    order = np.lexsort((np.arange(GPT2_V), -counts))   # -count primary, id secondary
    keep = order[:V - 1]
    lut = np.zeros(GPT2_V, dtype=np.uint16)            # 0 = UNK for everything else
    lut[keep] = np.arange(1, V, dtype=np.uint16)
    return lut, keep


def unk_stats(mapped):
    """Per-token and per-sequence UNK statistics for one split."""
    is_unk = (mapped == 0)
    per_seq = is_unk.mean(axis=1)
    return {
        'n_seqs': int(mapped.shape[0]),
        'n_tokens': int(mapped.size),
        'unk_rate_per_token': round(float(is_unk.mean()), 6),
        'unk_rate_per_seq_mean': round(float(per_seq.mean()), 6),
        'unk_rate_per_seq_p50': round(float(np.percentile(per_seq, 50)), 6),
        'unk_rate_per_seq_p95': round(float(np.percentile(per_seq, 95)), 6),
        'unk_rate_per_seq_max': round(float(per_seq.max()), 6),
        'frac_seqs_with_any_unk': round(float((per_seq > 0).mean()), 6),
    }


def token_strings(keep):
    """(gpt2 id -> BPE token, decoded string) for every kept id; empty on failure
    (labelling is a convenience, never a correctness dependency)."""
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained('gpt2')
        toks = tk.convert_ids_to_tokens([int(i) for i in keep])
        dec = [tk.decode([int(i)]) for i in keep]
        return toks, dec
    except Exception as ex:                                # noqa: BLE001
        print(f'  (token strings unavailable: {ex})', flush=True)
        return [], []


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build(V, arr, counts):
    out = f'{HERE}/tf_corpus_v{V}'
    os.makedirs(out, exist_ok=True)
    lut, keep = build_lut(counts, V)
    print(f'== V={V}: coverage of the train split by the kept {V - 1} ids: '
          f'{counts[keep].sum() / counts.sum():.6f}', flush=True)

    man = {'vocab_size': V, 'unk_id': 0, 'seq_len': SEQ,
           'source': 'basis_aligned/qk_mdl/corpus_fresh/shard00..06.npy '
                     '(FineWeb sample-10BT, docs 45367..267574, gpt2 tokenizer)',
           'n_source_seqs': int(len(arr)), 'shard_seqs': SHARD_SEQS,
           'splits': {k: list(v) for k, v in SPLITS.items()},
           'vocab_selected_on': 'train split only (rows 0:240000)',
           'tie_break': 'descending count, ascending gpt2 id',
           'train_coverage_by_kept_ids': float(counts[keep].sum() / counts.sum()),
           'single_epoch': {'steps': TF_STEPS, 'batch': TF_BATCH,
                            'seqs': TF_STEPS * TF_BATCH,
                            'predicted_tokens': TF_STEPS * TF_BATCH * (SEQ - 1)},
           'splits_stats': {}, 'files': {}}

    for name, (lo, hi) in SPLITS.items():
        mapped = lut[arr[lo:hi].astype(np.int64)]          # uint16 out
        man['splits_stats'][name] = unk_stats(mapped)
        files = []
        for s, a in enumerate(range(0, len(mapped), SHARD_SEQS)):
            p = f'{out}/{name}{s:02d}.npy'
            np.save(p, mapped[a:a + SHARD_SEQS])
            files.append({'file': os.path.basename(p),
                          'rows': int(min(SHARD_SEQS, len(mapped) - a)),
                          'sha256': sha256(p)})
        man['files'][name] = files
        st = man['splits_stats'][name]
        print(f'   {name:6s} {st["n_seqs"]:7d} seqs  UNK/token '
              f'{st["unk_rate_per_token"]:.4f}  UNK/seq mean '
              f'{st["unk_rate_per_seq_mean"]:.4f} p95 '
              f'{st["unk_rate_per_seq_p95"]:.4f}', flush=True)
        del mapped

    toks, dec = token_strings(keep)
    vocab = {'vocab_size': V, 'unk_id': 0,
             'gpt2_ids': [0] + [int(i) for i in keep],     # index = new id
             'counts_train': [0] + [int(counts[i]) for i in keep],
             'tokens': ['<UNK>'] + toks, 'decoded': ['<UNK>'] + dec}
    json.dump(vocab, open(f'{HERE}/tf_vocab_{V}.json', 'w'))
    np.save(f'{out}/lut_gpt2_to_new.npy', lut)
    man['vocab_file'] = f'tf_vocab_{V}.json'
    man['lut_file'] = 'lut_gpt2_to_new.npy'
    json.dump(man, open(f'{out}/MANIFEST.json', 'w'), indent=2)
    return man


# ---------------- loaders used by tf_train / tf_fold ----------------
def load_split(V, name, n=None, root=None):
    """Concatenated int64 rows of one split, in shard order (prefix of n rows)."""
    root = root or f'{HERE}/tf_corpus_v{V}'
    man = json.load(open(f'{root}/MANIFEST.json'))
    rows, got = [], 0
    for f in man['files'][name]:
        a = np.load(f'{root}/{f["file"]}', mmap_mode='r')
        take = len(a) if n is None else min(len(a), n - got)
        if take > 0:
            rows.append(np.asarray(a[:take]))
            got += take
        if n is not None and got >= n:
            break
    if n is not None:
        assert got == n, f'{name} split too small: {got} < {n}'
    return np.concatenate(rows).astype(np.int64)


def load_vocab(V):
    return json.load(open(f'{HERE}/tf_vocab_{V}.json'))


def label(vocab, new_id):
    """Human-readable label for a reduced-vocabulary id (for table printing)."""
    d = vocab['decoded'][new_id] if new_id < len(vocab['decoded']) else ''
    return f'{new_id}:{d!r}'


if __name__ == '__main__':
    t0 = time.time()
    vs = [int(a) for a in sys.argv[1:]] or [4096, 8192]
    print(f'loading source shards from {SRC} ...', flush=True)
    arr = load_source()
    print(f'source {arr.shape} {arr.dtype} ({time.time() - t0:.0f}s)', flush=True)
    lo, hi = SPLITS['train']
    print('counting GPT-2 id frequencies over the TRAIN split ...', flush=True)
    counts = count_ids(arr, lo, hi)
    print(f'  {int((counts > 0).sum())} distinct ids in train '
          f'({time.time() - t0:.0f}s)', flush=True)
    for V in vs:
        build(V, arr, counts)
    print(f'done in {time.time() - t0:.0f}s', flush=True)
