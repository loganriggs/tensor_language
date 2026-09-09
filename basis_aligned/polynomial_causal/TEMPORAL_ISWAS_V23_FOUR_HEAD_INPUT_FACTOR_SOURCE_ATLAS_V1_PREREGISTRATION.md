# Temporal is/was v23 four-head input factor/source atlas v1

**Status:** preregistered before model execution

## Circuit decision

The existing v23 circuit identifies four causal head-output writers but not how the input tokens
produce those writes. This experiment asks whether their output change is attributable to temporal
cue tokens, unchanged prefix tokens, or the matched reporter suffix, and whether it is carried by a
proper subset of the exact Q1/K1/Q2/K2/effective-value factors.

It targets computational specification and input-side circuit localization. It does not fit a rank,
subspace, or activation decoder and cannot by itself establish a directed upstream edge.

## Frozen population and sites

- all 64 v23 rows: 16 each A1, A2, P, and C;
- target scoring on the already frozen 30 jointly native-capable A1/A2 rows;
- controls unfiltered;
- sites `L8H1`, `L9H1`, `L9H4`, `L11H3`;
- every query position through each row's semantic endpoint;
- no v24 access.

For every row, source tokens are partitioned without model outcomes into:

1. `changed`: base and donor token IDs differ;
2. `unchanged_prefix`: equal IDs before the longest common suffix;
3. `matched_suffix`: the longest exact common suffix ending at the reporter endpoint.

The partition must cover every real prefix token exactly once and no padded token.

## Exact factor game

For one head, query $$t$$, and source $$s$$, the pre-output-projection contribution is

$$
z_{t,s}
=\left(\frac{q_t\cdot k_s}{128}\right)
 \left(\frac{q^{(2)}_t\cdot k^{(2)}_s}{128}\right)v_s.
$$

The runner evaluates all $$2^5=32$$ base/donor mixtures of
`q`, `k`, `q2`, `k2`, and the effective value `v`, simultaneously at all four sites. It also
evaluates all $$2^3=8$$ termwise source-role coalitions, reusing the empty and full arms from the
factor game. Möbius/Shapley accounting is exact for both finite games.

Every mixed result is installed as an **absolute** head-output clamp. Additive base-derived deltas
are forbidden because earlier patched heads change the live input to later heads.

## Bars and predictions

- raw head reconstruction/mixture closure: maximum absolute error `1e-4`;
- replay of every reported target/control scalar from the immutable v23 four-head union: `2e-3`;
- target recovery `.50`, cosine `.90`, direction fraction `.90`;
- P and C behavior leakage ratio at most `.15`;
- distributed factor allocation: at least three factors have Shapley allocation `.05` and positive
  allocation in both frozen reporter halves;
- changed-token dominance: `changed` is the largest source-role Shapley allocation, at least `.25`,
  positive in both halves, and its singleton arm passes the target/control bars.

Registered predictions:

1. authority, population, partition, finiteness, enumeration, and exact price pass;
2. factor/source closure and the prior four-head union replay pass;
3. changed temporal tokens dominate the source-role game;
4. at least one proper Q/K/Q2/K2/value subset passes the selective target bars;
5. at least three of the five input factors receive split-stable positive credit above `.05`.

Predictions 3-5 are scientific and may fail without invalidating the instrument. If prediction 3
fails, the strongest source claim is distributed or suffix/prefix dominated. If prediction 4 fails,
the earlier v15 aggregate null generalizes: the writer computation is coupled at this resolution.
If prediction 5 fails, the factor computation is concentrated rather than distributed.

## Literal price

- one checkpoint load;
- 47 forwards and `47 * 64 = 3,008` sequence evaluations;
- zero backward passes, optimizer updates, or fitted parameters;
- eight capture forwards, two native reference forwards, 31 nonempty factor arms, and six proper
  source-role arms; empty/full source arms reuse the factor game.

The result is invalid if the observed price differs, if any factor/source/full-head closure fails,
or if the all-factor/all-role arm does not replay the immutable v23 union.
