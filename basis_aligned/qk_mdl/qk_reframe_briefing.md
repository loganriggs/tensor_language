# Reframe briefing: what exactly we're confused about, with real examples

*(Tick 228. All examples are real tokens from the held-out FineWeb audit set.)*

## 1. The confusion, localized to one box

Everything up to layer-1's pattern is explained except **one map**: from the 16-token
window to the context corrections of layer-1's QK factors (the 576 adapter
coordinates). Of that map: 29% is explicit second-order token interactions (the
pairwise object); ~70% is unexplained. The unexplained part is further localized:
key-side channels of the lexical/content heads (k1 of layer-1 heads 3, 5, 7; k2 of
head 1), on one text class (below), at attention offsets 0–2.

## 2. Real examples: understood versus not

**Understood, with causal verification** (the static tables + archetype story — 99%
of the pattern route's function):

| context (real tokens) | target | why we understand it |
|---|---|---|
| "…Facebook's Graph Search is in search" | " of" | l0-h8 scaffold archetype; ablating that one channel breaks exactly this (+2.5 nats) |
| "…with Mack Brooks Exhibitions Ltd." | " of" | l0-h6 delimiter channel, same causal test (+6.1) |
| "…in one large-format playing edition, are" | " Pagan" | l0-h3 determiner/copula context (+4.0 when ablated) |
| any "the/a/an → noun", punctuation, quote-pairing, auxiliaries | — | validated archetype matches, tables suffice |

**Not understood** (worst positions under our best explicit context object):

| context (real tokens) | target | ΔCE |
|---|---|---|
| "…Last month, Charlie gave Lindsay" | " L" (→"Lohan") | 6.2 |
| "…I'm not sure if Matvich" | "uk" (→"Matvichuk") | 5.8 |
| "…Southridge Minerals, Inc. (SR" | "GE" (ticker) | 4.9 |
| "…Angel Falls, Venezuela⏎• Nazca" | " lines" | 4.6 |
| "…at Act II, in Beneath a Granite" | " Sky" (title) | 4.4 |
| "…in support of Bradley" | " Manning" | 3.7 |

**The semantic label for the missing 70%: entity-identity carry.** Every hard case is
a multi-token *name* mid-spelling — person, ticker, title, place — where the needed
information is *which specific entity* the last few tokens began. This is sharper
than "subword continuation": completing "Matvich→uk" requires the exact rare string;
completing "Lindsay→ L(ohan)" requires entity knowledge keyed by the exact preceding
name. Coarse embedding codes destroy exactly this (why order-1 and our 96-dim pair
codes both fail here), and it is combinatorial in the window (why bigram tables fail).

## 3. The computational budget, and the interactions we have NOT accounted for

At block 0 the model can only compute functions of: `emb(tok_i)` + one attention pass
over the window + one bilinear MLP. The MLP input is `h = emb-part + attn0-part`, and
because the layer is bilinear, its output splits EXACTLY into three blocks (weight ×
unigram shares from §7f):

| block | token-order | wt-space share | our coverage |
|---|---|---|---|
| emb × emb | 2 | 84% | **this is what the pairwise object fits** (and the token tables) |
| emb × attn0 | ~4 (P0 is degree-2) | ~3% | **not accounted** |
| attn0 × attn0 | ~6 | ~11% | **not accounted — yes, attention-with-attention** |

Two caveats make the small-looking shares matter: the weight-space shares assume
independent sampling (realized, correlated shares can differ), and the *residual we
fail on* is presumably concentrated in exactly these blocks, since the emb×emb block
is what we already capture. The raw-token polynomial ladder could never represent
these blocks compactly because they are **P0-weighted** — but P0 is not unknown! It is
the thing we spent forty ticks decomposing. Which suggests the reframe:

## 4. Proposed reframe: compose the circuit out of its own named variables

Instead of raw embedding codes at offsets, build the feature set from the validated
layer-0 ledger: per layer-0 head and archetype r, the **archetype activation**
g_r(i) = ⟨attn0_h(i), w_r⟩ (pattern-weighted class content, computable from tables +
the P0 kernel — all objects already in the description ledger, so they cost no new
bits). Then fit the interface as bilinear in [embedding code of tok_i] ×
[archetype activations] (the emb×attn0 block, in named coordinates) plus
[g × g] terms (attn0×attn0). Every feature has a name ("head 3's {the}-weighted
content", "head 1's fragment-weighted content"); the fitted maps are small; and the
entity-carry hypothesis becomes testable: the hard cases should load on the
fragment/name archetype activations specifically.

## 5. Success metric, narrowed per Logan

Object = the interface map only (no reconstruction of MLP or QK weight structure).
- C_interface = 0.0515 (static → exact on the pattern route).
- F = fraction recovered on the full audit, as always.
- L = bits of the MAP ONLY; upstream l0 objects are charged once in the global ledger,
  not again here (they are reused variables, which is the point).
- Current row: pairwise 29% at 14 Mb. First reframe target: beat 29% with archetype-
  feature bilinears; stretch target: the weight-referencing generators' 45%.
