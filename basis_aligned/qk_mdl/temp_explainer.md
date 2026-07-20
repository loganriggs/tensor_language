# Working explainer: the layer-0 story, and why it changes above layer 0

A running document to answer Logan's questions in plain language, one at a time. Terms
spelled out; the terse file-labels given in parentheses where useful.

---

## The setup: why layer 0 is special

A transformer layer reads from the *residual stream* — the running sum of vectors that
every layer adds to. At **layer 0**, the residual stream is just the token embeddings
(plus position). That means every quantity layer 0 computes is a fixed function of the
**token identities and their positions** — nothing contextual has entered yet. Because
of that, we can fold the embedding matrix directly into layer 0's weight matrices and
get exact vocabulary-indexed tables: for each token in the 50,257-word vocabulary we can
write down, from the weights alone, what layer 0 does with it. No estimation, no data —
the combinatorics don't blow up because everything is a function of *one* token at a
time (or, for attention, an ordered *pair* of tokens plus their distance, which is still
tractable).

Layer 0 has three circuits, and we reduce all three. Here is each one.

---

## 1. The selection circuit (labelled QK): reduced by clustering

**What it computes.** Attention decides *where each token looks*. For a query position
and a key position, it computes a score — high score means "attend here." At layer 0 the
score is a fixed function of the query token and the key token (and their separation).
So for each attention head we can build two tables, one for the query side and one for
the key side, each with one row per vocabulary word. The score between two positions is
essentially the dot product of the query word's row with the key word's row.

**How we reduce it.** We run k-means clustering on those rows — grouping the 50,257 words
into, say, 256 classes such that words in the same class have nearly the same
query/key behaviour. We then replace each word's row with its class's average row. Now
the attention scores depend only on *which class* the query and key words belong to, not
their exact identity. This costs almost nothing in loss (at 256 classes it is roughly
free, and after a light retraining it is actually slightly *better* than the original).

**Why it works here:** selection is a coarse operation. The model doesn't need to know
the query is exactly the word "apple" — it needs to know "apple" is a common noun, and
common nouns of that kind attend the same way. The clustering only affects the **scores**
(the attention pattern), not the content that gets moved.

---

## 2. The content circuit (labelled OV): reduced by a sparse dictionary

**What it computes.** Once attention has decided where to look, the *content* (value)
circuit says *what to copy* from the attended position into the output. At layer 0, each
token again has a fixed content vector we can fold from the weights — one 128-dimensional
vector per word per head.

**Why clustering fails here.** We tried the same k-means trick and it fell apart:
grouping words into classes for their *content* is destructive (256 classes cost +1.38
nats, versus roughly free for selection). The reason is that carrying a word forward
requires its **fine identity** — "apple" and "apricot" attend similarly but you cannot
substitute one for the other when you are actually transporting the word's meaning. Even
retraining only recovers about a third of the damage. This is the program's central
dichotomy: **selection needs classes, carriage needs identity.**

**How the sparse dictionary actually works.** Instead of replacing each word's content
vector with a single class average, we express it as a **sparse combination** of shared
building blocks:

1. Learn a shared *dictionary* of, say, 512 atom-vectors (per head).
2. For each word, represent its content vector as a weighted sum of just **16** of those
   512 atoms — each with its own signed coefficient. So the description of word *t* is:
   "take atom #37 times +0.8, plus atom #182 times −0.4, plus … (16 terms total)."
3. Store, per word, only which 16 atoms and their 16 coefficients.

The key difference from clustering: clustering says "you *are* one of 256 things"; the
dictionary says "you are a specific *mixture* of 16-out-of-512 things." A mixture of 16
signed atoms can express far finer distinctions than a single class label, so it
preserves the identity that content transport needs. At 512 atoms × 16 coefficients this
lands at +0.034 nats before retraining and **−0.019 after** (better than the original) —
whereas hard classing bottoms out around +0.57 even with retraining. (Files:
`results/07_ov_blocks.md`, tables "Sparse coding rescues content.")

---

## 3. The bilinear feed-forward block: yes, we reduce it too

**What it computes and why it splits cleanly.** Each layer's feed-forward block in this
model is *bilinear*: its hidden activation is `(Left · x) ⊙ (Right · x)` — two linear
projections of the input, multiplied together element-wise. Because the input `x` at
layer 0 is the sum of the token embedding (`e`) and the attention output (`a`), the
multiplication expands **exactly** into three interaction blocks:

    (Left·x) ⊙ (Right·x) = [Le⊙Re]  +  [Le⊙Ra + La⊙Re]  +  [La⊙Ra]
                            self          cross              pair

- **self** — the token embedding interacting with itself (importance +1.29 nats)
- **cross** — the current token interacting with what attention brought in (+0.84)
- **pair** — the attention output interacting with itself (+0.19)

The split is exact to about one part in ten million (a numerical gate we check before
trusting it). We reduce each block by clustering its two input sides independently, and
every block tolerates roughly 256–1024 classes per side (self at 256 classes: +0.097;
cross sides: +0.043 and +0.055; pair at 256: +0.058). So the answer to "do we interpret
and reduce layer 0's bilinear block?" is **yes** — and the finding is that every
*interaction* in the block is class-tolerant, unlike content transport. Full scoreboard
in `results/07_ov_blocks.md` ("The complete MLP-0 decomposition").

**Layer-0 summary.** Every part of layer 0 compresses to *better than the original* once
you use the right tool for each: selection by classes (−0.039), content by sparse
dictionary (−0.019), feed-forward interactions by classes (+0.022). Layer 0 is
genuinely, fully understood.

---

## 4. Why token-conditional means appear above layer 0 (and how they are computed)

**The problem.** Everything above only works because layer 0's inputs are token-
determined, so we could fold exact tables from the weights. At **layer 1 and above** the
input is the residual stream *after* layer 0 has run — which depends on the whole
preceding context, not just the current token. There is no exact table to fold from the
weights anymore. If we still want a token-indexed description, we have to **measure it
from data** instead of deriving it from weights.

**What a token-conditional mean is.** For any quantity the model computes at some layer —
call it `z` at a given position — we ask: *on average, what is `z` when the current token
is this particular word?* Formally, the table row for word *t* is the average of `z` over
every position in a corpus where the current token happens to be *t*:

    table[t]  =  average of  z(position)  over all positions whose token is t

Concretely, to compute it we:
1. Run the model over a large chunk of text (a few hundred thousand tokens).
2. At the layer of interest, capture the quantity `z` at every position (for example, a
   layer-1 attention query vector, taken at the same normalization point where the
   weights would have folded).
3. Add each captured vector into the bucket for its current token, and count.
4. Divide each bucket by its count — that average is the token's table row.
5. Renormalize the rows to the natural scale the model uses at that point (this
   "gauge" matters — skipping it triples the cost).

Words never seen fall back to the global average. The whole set of these tables per
stream is what the later machinery calls "cond-mean tables."

**What this measures — and its honest cost.** A token-conditional mean captures the part
of `z` that is a pure function of the current token, and *averages away* everything
contextual. The surprising empirical result is how much that captures: for **selection**
at higher layers, replacing the live quantity with its token-conditional mean is nearly
free — attention above layer 0 turns out to be almost as token-determined as at layer 0,
just no longer derivable from weights alone. The part it throws away — the genuinely
contextual residue — is exactly what the "live window" in the later windowed architecture
keeps. And because these tables are *measured*, not folded, we report their estimation-
data cost (how many tokens we averaged over) alongside their bit cost, and never mix the
two.

**One-line contrast with layer 0:** at layer 0 the token→quantity map is *exact and
free* (it is the weights). Above layer 0 the token→quantity map is *estimated and lossy*
(it is a corpus average), and the loss is the contextual part — small for selection,
large for the two attention heads and top feed-forward directions we later isolate.

---

*(More questions from Logan to be appended here as they come.)*

---

## 5. What the dictionary is, exactly — and what a forward pass looks like

**The dictionary (per attention head).** For head *h*, we fold the embedding into the
value projection to get a table of content vectors, one 128-dimensional vector per
vocabulary word: call it `VT[t, h]`. The dictionary for that head is three things:

- `atoms` — a matrix of `n` unit-length vectors in 128 dimensions (e.g. n = 512). These
  are the shared building blocks.
- `bias` — one 128-dimensional vector, the average content.
- `encoder` — a matrix used only to *choose and weight* atoms (learned; it need not equal
  the atoms themselves).

To encode word *t*: project its (bias-subtracted) content onto the encoder, keep the `k`
largest-magnitude coordinates (the "top-k"), and store *which* k atoms and their signed
coefficients. So word *t* is described by `k` integer atom-indices and `k` real
coefficients. Reconstruction is: `bias + Σ (coefficient × atom)` over those k atoms.

```python
# ENCODE the whole vocabulary's content for one head (offline, once)
z = (VT_h - bias) @ encoder.T            # (vocab, n) raw coefficients
vals, idx = z.abs().topk(k, dim=1)       # (vocab, k) which atoms
coeff = torch.gather(z, 1, idx)          # (vocab, k) signed coefficients
# stored: idx (k int per word) + coeff (k float per word) + atoms + bias

# RECONSTRUCT the value table from the sparse code
VT_hat = bias + (coeff.unsqueeze(-1) * atoms[idx]).sum(dim=1)   # (vocab, 128)
```

**How it enters a forward pass — yes, it is essentially an indexing operation.** In the
live model, layer-0 attention computes each token's value vector by running the value
projection on the token's embedding. The reduction replaces that with a *table lookup*:
the value vector for token *t* is just row *t* of the reconstructed table. Because the
reconstructed table is itself defined by the sparse code, you can either (a) precompute
the whole `VT_hat` table and index into it, or (b) index the per-token code directly and
sum the k atoms on the fly — mathematically identical. In the actual audit code the
attention block's only change is one line:

```python
# LIVE model:
v = a.c_v(h).view(B, T, NH, HD)          # value = projection of the residual

# REDUCED (layer 0 only): value = sparse-dictionary table lookup by token id
v = VT_hat[token_ids]                     # (batch, seq, heads, 128) — pure gather
```

Everything downstream — the attention pattern, the mix, the rest of the layers — is
untouched. So "how is the decomposition included?" — the decomposition *defines a table*,
and the forward pass *indexes that table by the current token* instead of recomputing the
value from weights. The embedding never appears explicitly at layer 0 anymore; it has been
folded into (and then compressed inside) the table.

### Variants we are measuring (Logan's requests)

**Batch-top-k.** Per-token top-k forces *exactly* k atoms on every word — wasteful for
easy words, too tight for hard ones. Batch-top-k instead keeps the largest `k × vocab`
coefficients across the *entire* code matrix at once, so k is only the *average*: common
easy words spend fewer atoms, rare hard words spend more, at the same total budget.

```python
z = (VT_h - bias) @ encoder.T                 # (vocab, n)
nnz = k_avg * vocab                           # total budget across all words
thresh = z.abs().reshape(-1).topk(nnz).values.min()
z_sparse = z * (z.abs() >= thresh)            # flexible per-word sparsity
VT_hat = bias + z_sparse @ atoms
```

**Routed / block-sparse.** Instead of one big shared dictionary, cluster the vocabulary
into groups (by embedding similarity) and give each group its *own* small dictionary and
its own sparsity — your picture of "some words use 8-of-64, others 8-of-a-different-128."
Words route to their group's dictionary; only that group's atoms are candidates.

```python
group = kmeans_labels(embeddings)             # (vocab,) which group each word is in
for g in range(num_groups):
    ids = (group == g).nonzero()
    atoms_g, bias_g, enc_g = train_dict(VT_h[ids], n_g[g], k_g[g])   # own dict per group
    VT_hat[ids] = encode_reconstruct(VT_h[ids], atoms_g, bias_g, enc_g, k_g[g])
# bits: sum over groups of (atoms_g) + per-word codes + a small group-id per word
```

The bet: specialized small dictionaries per word-family beat one general large dictionary
at equal total bits (a number-words dictionary, a name-prefix dictionary, and so on). The
sweep numbers for all three schemes are appended below once the run lands.

### Sweep results (all three schemes, layer-0 content tables)

Reconstruction-fit (not retrained), same 1200-step budget per dictionary so the
*comparison* is fair even though the absolute numbers sit above the well-trained anchor
(the earlier n=512, k=16 result was +0.034 after 3000 steps; here it is +0.072 after
1200). Loss shown as cross-entropy increase over the live model; "megabits" is the
structural description size.

| scheme | sparsity | cross-entropy increase | size |
|---|---|---|---|
| per-token top-k (one dict of 512) | k=4 | +0.277 | 93 Mbit |
| | k=8 | +0.218 | 167 |
| | k=16 | +0.072 | 316 |
| | k=32 | +0.001 | 613 |
| batch-top-k (one dict of 512) | avg 4 | +0.413 | 93 |
| | avg 8 | +0.188 | 167 |
| | avg 16 | +0.064 | 316 |
| | avg 32 | +0.015 | 613 |
| **routed / block-sparse (8 groups)** | 8-of-128 each | **+0.134** | 179 |
| | 8-of-(64…160, adaptive) | **+0.123** | 189 |

**Reading the three schemes.**

- **Batch-top-k vs per-token:** giving words a *flexible* atom budget (same average,
  spent where it is needed) helps once the budget is comfortable — at average 8 and 16 it
  beats fixed per-token top-k (+0.188 vs +0.218; +0.064 vs +0.072). But at the very tight
  budget of 4 it is *worse* (+0.413 vs +0.277): when the total is that small, the flexible
  scheme starves many words to almost nothing. Flexibility helps only when there is slack
  to reallocate.

- **Routed / block-sparse is the clear winner at its budget.** Eight specialized small
  dictionaries (each word family gets its own) reach +0.123–0.134 at ~180 megabits, versus
  +0.19–0.22 for a single shared dictionary at the *same or larger* size. Your intuition
  holds: a name-prefix dictionary, a number dictionary, a word-fragment dictionary and so
  on each capture their family more efficiently than one general dictionary trying to
  serve all of them. The adaptive version (bigger dictionaries for bigger/harder word
  families) edges out the uniform one at a small extra cost. Notably the routed
  dictionaries were trained even *less* (800 steps each) yet still win — the specialization
  more than compensates.

**Bottom line for your question:** the single shared 16-of-512 dictionary was the first
thing that worked, but it is not the efficient frontier. Routing content into per-family
dictionaries is meaningfully better at equal bits, and is the natural next form for the
content reduction. (Files: `ov_dict_variants.py`, `ov_dict_variants.json`.)
