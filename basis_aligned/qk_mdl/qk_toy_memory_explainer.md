# Toy models of the memory-MDL claim

*(Tick 233; all numbers computed, code in `qk_toy_memory.py`. Claim under test: "the
MLP's parametric encoding of long-tail entity memory is near the description-length
frontier — no small explicit object exists." The toys escalate toward the real case;
the honest wrinkle in Toy B and its refinement are the heart of the argument.)*

## Toy A — parametric storage of random pairs is table-like, with overhead

A bilinear net (Left/Right/Down, the real block's shape) memorizes N random
key→value pairs (values uniform over 1024; information floor 10 bits/pair). Measured
capacity: perfect recall at ~320 fp32-bits per pair, collapsing by 160 (60k params
hold 6,000 pairs; 240k params hold 24,000 — capacity ≈ params/10). So in fp32 the
parametric form pays ~32× the floor, ~8× if int8-quantized: **the same order as an
explicit table**. Parameters are a legitimate, if not optimal, association store —
"it's just weights" is not evidence of waste.

## Toy B — raw Zipf: small tables DO capture the head (the honest wrinkle)

200k pairs under Zipf(1) exposure: 50% of the total CE gain needs only the top 335
pairs; 75% needs 4% of pairs. A naive reading says small tables are great — and they
are, **for the head**. This is exactly what our 3M/30M-token datastores did: covered
the head, gained nothing — because:

## Toy B′ — the real case starts PAST the head (the refinement that matters)

The model's tables + named basis already cover the head. The residual lives in the
tail, and tail arithmetic is brutal. With Zipf(1), cumulative gain ≈ ln(M)/ln(N).
Take the realistic entity-pair space N ≈ 10⁸ and a baseline already at 75% coverage
(M₀ = N^0.75 = 10⁶ pairs). Then:

- **halving the residual** (75% → 87.5%) needs M = N^0.875 = **10⁷ pairs ≈ 400 Mbit**
  of key–value table —
- which is **the same order as the MLP's own 680 Mbit.**

That is the claim, toy-verified: once the head is free, buying the next half of the
tail explicitly costs as much as the parametric organ that already stores it. The
datastore nulls (ticks 231–232: 3×10⁶ and 3×10⁷ tokens ≈ M ≪ 10⁶ useful tail pairs)
were foreordained by this arithmetic.

## Toy C — rules + exceptions reproduces the program's observed shape

Half the keys follow a small shared rule; half are exceptions; Zipf exposure. The
measured explicit-object frontier:

| object | bits | fraction of gain |
|---|---|---|
| the rule alone | 2.1 Mb | **45%** |
| + top-1,000 exceptions | 2.12 Mb | 85% |
| + top-20,000 exceptions | 2.6 Mb | 98% |

The knee at the rule (~45–50% at tiny bits, then linear-in-table) is exactly the real
program's curve: **named basis = the rule (51%)**; the remainder = the exception
table, whose real-world size (Toy B′) is MLP-scale. The toy's exceptions are cheap to
finish only because its universe is 60k pairs; scale the exception space to 10⁸ and
Toy C's flat segment becomes the real one.

## The mapping, explicitly

| toy element | real-circuit element |
|---|---|
| the rule g | token tables + archetype activations (named basis, 51%) |
| head of the Zipf table | common n-grams (what our datastores covered — no gain) |
| tail exceptions | first-mention entity continuations (Lindsay→Lohan, Matvich→uk) |
| parametric memorizer at ~320 bits/pair | the block-0 MLP (~680 Mbit) |
| Toy B′ residual-halving cost ≈ 400 Mbit | why no small explicit object exists |

## Next refinements available (each one step more realistic)

1. **Fuzzy keys**: exceptions sharing morphology (soft keys) — tests whether soft-key
   retrieval could beat the table arithmetic (the one live alternative from tick 232).
2. **Exposure-matched training**: memorizer trained under the same Zipf it is scored
   on — capacity shifts toward the head, quantifying how much tail the real MLP can
   actually hold (an upper bound on what remains recoverable at all).
3. **Composed keys**: keys built from token windows through an attention layer —
   closing the loop to the actual architecture.
