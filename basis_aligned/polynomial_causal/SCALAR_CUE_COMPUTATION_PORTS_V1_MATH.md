# Where head9's regional cue contrast enters the computation

12September2026,23:00UTC. **On the cached cue pairs, changing source keys and values while holding the query fixed reproduces the head9.8 scalar cue contrast within5.5–7.9%.** The source value is the largest contributor under a symmetric allocation, but key–value interactions prevent a baseline-independent value-only explanation. These are local computational tests, not native final-logit mediation results.

## Scope and prior-art check

Use the frozen rank64 reflection-even head9.8 component and72pristine states from the earlier city/nationality/style cue panel. No weights or states are fitted. Each British/American pair differs at one token and has matching length/positions. The earlier source-city removal test concerned a different compiled mixed-token component and already warned that cue influence can reside in contextual descendants rather than the city position alone. This experiment swaps the **entire source input group**, not just the literal cue position.

The managed six-mixed-band addition experiment remains separate. Its immutable job is queued behind the live peer671run; it has not been restarted. An existing single-band diagnostic shows different regional/newline profiles, but additive estimates near the preservation boundary do not replace actual union interventions. [Single-band signatures](MIXED_BAND_ATOM_SIGNATURES_V1.json).

## Query versus source keys and values

Write $f(Q,S)$ for the selected head9 scalar at the final query position. $Q$ supplies both QK query factors; $S$ supplies both source-key factors and the source value, with all native projected normalizers and rotary positions retained. Both arrays come from pristine cached model states. This does not assign separate tasks to QK1 versus QK2.

Let $U,A$ denote British/American cue contexts. Evaluate all four hybrids $f(U,U),f(A,U),f(U,A),f(A,A)$. The target is the native scalar contrast

$$
\Delta=f(U,U)-f(A,A).
$$

Its symmetric computational allocation is

$$
\Delta_Q=\tfrac12[f(U,U)-f(A,U)+f(U,A)-f(A,A)],
$$

$$
\Delta_S=\tfrac12[f(U,U)-f(U,A)+f(A,U)-f(A,A)],
\qquad\Delta_Q+\Delta_S=\Delta.
$$

Native hybrid endpoints agree with cached scalars within3.06e-7relative; accounting error is8.53e-17. The direct source-only comparison uses $f(A,U)-f(A,A)$; query-only uses $f(U,A)-f(A,A)$.

| Cue family | Source-only error | Query-only error | Query aligned fraction | Source aligned fraction |
|---|---:|---:|---:|---:|
| City | 7.90% | 111.36% | −7.89% | 107.89% |
| Nationality | 5.46% | 102.94% | −3.79% | 103.79% |
| Style | 7.13% | 101.91% | −4.14% | 104.14% |

An aligned fraction is $\langle\Delta_Q,\Delta\rangle/\|\Delta\|^2$, evaluated across paired examples. It is not a probability or fraction of variance. Opposing contributions explain values outside0–100%. Source-only passes its20%error bar in everyfamily; query-only fails. [Result](QUERY_SOURCE_CUE_PORTS_V1_RESULT.json).

## Within the source group: joint keys versus value

Hold the query at its American state and write $g(K,V)$ for the scalar with independently supplied joint source keys and value. Again evaluate four British/American hybrids. The target here is the **source-only contrast** from the preceding experiment, not the full native contrast. Tied key/value corners replay the previous source corners exactly.

| Cue family | Symmetric value aligned fraction | Key-only error | Value-only error, American keys |
|---|---:|---:|---:|
| City | 95.68% | 112.34% | 22.91% |
| Nationality | 97.27% | 104.78% | 12.08% |
| Style | 92.78% | 103.48% | 22.95% |

Instrument/accounting pass, but both general20%sufficiency criteria fail: value-only misses oncity/style and key-only misses everywhere. The independently evaluated four-corner key–value interaction has17.50–35.99%of the source-contrast norm. [Result](SOURCE_KEY_VALUE_CUE_PORTS_V1_RESULT.json).

For $g=\Gamma(K)V$, the finite change is

$$
\Delta g=\Gamma_A\Delta V+\Delta\Gamma V_A+\Delta\Gamma\Delta V.
$$

The symmetric value term uses $\tfrac12(\Gamma_U+\Gamma_A)\Delta V$ and therefore receives half of the interaction. A large symmetric value allocation does **not** prove that fixed-key value substitution suffices.

## Executed red-team of the value-only miss

Changing the fixed-key baseline to British gives value-only errors13.83%,5.44%,3.80%. Averaging the two value changes gives errors5.86%,3.36%,9.83%. Those alternatives use both observed contexts; they are diagnostics, not an improved autonomous predictor. Baseline orientation changes the value-only estimate by17.50–35.99%of the target norm. [Baseline diagnostic](SOURCE_KEY_VALUE_BASELINE_V1_DIAGNOSTIC.json).

The original failed criterion stays failed. Its strongest counter-review is that source value clearly carries the main cue-dependent change under the explicit symmetric convention, while key changes modulate its expression. It would be too pessimistic to conclude that no compact value generator could help; it would be too optimistic to delete the key dependency.

The next input-generation target is therefore the **source value reading**, together with its key-dependent modulation. Existing backward-folded MLP8 value-generator results must be reused before any new factor fit. These local hybrids have not yet established signed native logit mediation, broad OOD behavior, or independent generation of their pristine inputs.
