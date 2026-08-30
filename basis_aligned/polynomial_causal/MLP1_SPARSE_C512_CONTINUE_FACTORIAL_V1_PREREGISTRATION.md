# MLP1 sparse-Down × MLP0-C512 × MLP2-CONTINUE512 factorial v1

**Frozen before row selection, model access, or training:** 2026-08-30 UTC

## Question

Does a genuinely executable sparse MLP1 `Down` program preserve its standalone CE
benefit and compose with the already frozen MLP0-C512 and MLP2-CONTINUE512 programs?

Existing evidence does not answer this.  Earlier C512 factorials leave MLP1 native;
earlier three-layer cubes use exact-write or rank-64 oracle corrections rather than an
executable MLP1 replacement.  Historical MLP1 weight-action TopK fits are positive but
did not serialize a reusable program and did not cross C512 with CONTINUE512.

The tempting compiler-v2.1 `B_l6_r64` candidate is explicitly excluded.  Its rank-64
correction requires a frozen token-table/ridge producer, so the complete literal
program stores 60,707,648 reals (3.812 times native MLP1), not 153,920.  The frozen
deterministic receipt is `historical_mlp1_candidate_price_audit.json`.

## Fresh rows and role firewall

Freeze 288 registry-fresh FineWeb source documents beginning at ordered dataset document
index 122,000, with one 257-token row per document.  Split by document into three
disjoint 96-document roles:

- `FIT`: may train the MLP1 sparse program;
- `SELECT`: may choose one seed and check convergence/standalone gates;
- `FINAL`: one-shot factorial evaluation only.

The source-closed runner must publish the selected program bundle and a receipt before
loading the `FINAL` tensor.  No token is an independent replicate.  Score positions
64:256 while retaining the full 0:256 causal prefix.

## Frozen MLP1 program

Only MLP1's bias-free `Down` action is replaced.  Native MLP1 RMSNorm, `Left`, `Right`,
their 4,608 Hadamard products, and the separate native `Down_bias` remain exact.

For gate vector $g\in\mathbb R^{4608}$, fit

$$
\widehat D_1(g)=c + A\,\operatorname{TopK}_{32}(E g),
$$

where $E\in\mathbb R^{512\times4608}$ has unit-normalized rows, TopK keeps the 32
largest scores and applies ReLU exactly as in the earlier successful weight-action
assay, $A\in\mathbb R^{1152\times512}$, and $c$ is one explicit compressor intercept.
Use three seeds `[0,1,2]`, Adam, 2,400 steps, batch
size 1,024 token positions, learning rate `0.003` with cosine decay, and an evaluation
curve every 200 steps.  There is no input noise and no CE term.

Choose the seed with greatest final `SELECT` output $R^2$; break an exact tie by lower
seed.  CE is not a selection variable.  The selected bundle is frozen before `FINAL`.

## Price and execution

The program stores

$$
512(4608+1152)+1152=2,950,272
$$

floating constants, versus 5,308,416 in native MLP1 `Down`, a removal of 2,358,144
constants (`44.42%`) at that matrix.  Executed dense score products are
$512\times4608$; sparse decoding uses $32\times1152$ products.  TopK comparisons,
indices, and the intercept addition are reported separately.  Native Left/Right and
Hadamard products receive no compression credit.

## Pre-FINAL selection gates

All are descriptive except the first, which is required to admit the candidate to the
factorial:

1. **Executable standalone value:** selected `SELECT` CE recovery relative to zeroing
   the MLP1 Down action is at least 0.90.
2. **Convergence:** last-three-checkpoint $R^2$ range at most 0.01 and final $R^2$
   within 0.005 of the best checkpoint.
3. **Seed stability:** final `SELECT` $R^2$ standard deviation at most 0.02.

If gate 1 fails, publish a clean selection failure and do not open `FINAL`.

## One-shot factorial

Run all eight live sequential arms over:

$$
A\in\{\text{native MLP0},\text{C512}\},\quad
B\in\{\text{native MLP1},\text{sparse MLP1}\},\quad
C\in\{\text{native MLP2},\text{CONTINUE512}\}.
$$

Every downstream component remains native.  Each replacement must bypass its replaced
native `Down`/MLP call and preserve live sequential state, so B sees A's state and C
sees both upstream writes.  Add one `MLP1_DOWN_ZERO` control on the otherwise native
background to recompute FINAL standalone CE recovery.

Let $L_{abc,d}$ be source-document CE.  Publish all singleton effects, pairwise Möbius
terms, and

$$
I_{ABC}=L_{111}-L_{110}-L_{101}-L_{011}+L_{100}+L_{010}+L_{001}-L_{000}.
$$

Also publish the conditional cost of sparse MLP1 under each `(A,C)` background and the
change in the C512×MLP1 penalty when CONTINUE512 is installed.  Use 20,000 deterministic
source-document bootstrap draws, keeping all arms together, with percentile 95% CIs and
one max-absolute simultaneous band over the registered contrasts.

## Registered predictions and decisions

1. **Standalone transport:** FINAL sparse-MLP1 CE recovery is at least 0.90 and its
   95% lower bound is at least 0.85.
2. **C512 interaction:** the C512×sparse-MLP1 pairwise interaction is positive.  This
   tests whether independently compressed MLP1 amplifies C512's state shift.
3. **MLP2 compensation:** CONTINUE512 reduces that positive C512×MLP1 interaction; the
   reduction has positive point estimate.  CI sign is reported, not required.
4. **Composability:** the all-three arm's CE increase is no more than the sum of its
   three singleton increases plus 0.005 nat, and the simultaneous upper bound is no
   more than plus 0.010 nat.

No prediction is required to be true for the run to be informative.  The decision tree
is:

- if standalone transport fails, reject this MLP1 program before any semantic claim;
- if standalone passes but composability fails, retain it as a local compressor and
  target the measured interaction, not another isolated MLP fit;
- if both pass, serialize the three-component program and next test fresh OOD transport
  and selective edits at the same full price.

## Claim boundary

This experiment can establish or reject one executable early-layer composition.  It
does not make individual sparse atoms canonical or semantic, simplify Left/Right,
explain later MLPs, or move the strict whole-model ledger without the registered FINAL
and subsequent OOD/edit gates.
