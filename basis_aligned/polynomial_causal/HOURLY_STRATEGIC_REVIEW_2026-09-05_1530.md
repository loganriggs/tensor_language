# Hourly circuit-only review — 2026-09-05 15:30 UTC

## Controlling goal and decision

The controlling goal remains a reusable codebase that can produce hundreds of nonduplicated,
high-quality causal circuits and then use their shared structure to decompose the model. A useful
circuit must predict task behavior on held-out or OOD data, support causal interchange or selective
removal, expose a mechanism smaller than arbitrary native module boundaries, preserve unrelated
behavior, and reveal reuse or composition where possible. This hour did not open rank reduction,
quantization, activation-energy, variance, or activation-reconstruction work.

The main result is a causal chain substantially narrower than a whole attention head:

$$
\text{prior MLP writes}\;\longrightarrow\;V_{11}^{(h=3)}\hat{x}_{11,8}
\;\longrightarrow\;p_8u_8\;\longrightarrow\;\text{subject--verb agreement evidence}.
$$

On the licensed fresh held-out prompts, the current-state branch of the layer-11 head-3 subject
value carries the task, while its cached layer-0 value does not. Splitting the current-state input
shows that accumulated MLP writes from blocks 0--10 recover `0.773--0.913` of the full writer-family
effect. The embedding/skip path recovers `0.097--0.161`, prior attention writes recover
`0.014--0.066`, and broad-family interactions are small.

The next exact split found that the MLP contribution is not one isolated coarse depth group.
MLP0--3 recover about `0.04--0.06`, MLP4--7 about `0.22--0.27`, and MLP8--10 about `0.66--0.71`.
No singleton met the frozen all-cell `0.70` criterion, so none was promoted after seeing the data.
Their interactions are only `0.02--0.05`, and the observed MLP4--10 pair recovers `0.936--0.952`.
That pair is exploratory motivation for the next preregistration, not a retroactive held result.

## Rolling-hour terminals and serial time

From 14:28 through 15:28 UTC, eight scientific executions reached preserved terminals:

1. two countability native-capability attempts, both honest nulls;
2. one Task14 current-versus-cached engineering invalid;
3. one additive-scope selective causal-site screen;
4. the repaired Task14 current-versus-cached valid screen;
5. one Task14 upstream-writer engineering invalid;
6. the repaired upstream-writer valid screen; and
7. the valid Task14 three-way MLP depth-group factorial.

This is eight terminals in 60 minutes, or 7.5 serial wall minutes per terminal. The three Task14
valid causal localizations each used four forwards and finished in six or seven seconds. The GPU
was not the bottleneck. Most elapsed time was stable-file handoff, tests, claims, publication, and
repairing exact finite-precision bookkeeping.

## Engineering and scientific lessons

1. A mathematically exact residual decomposition need not be bitwise exact after regrouping many
   float32 additions. Both repaired experiments now record an explicit remainder with a frozen
   ownership convention and retain the original strict error bar; they do not hide the issue by
   loosening tolerance.
2. Wrapper experiments that inherit a scorer must repeat the frozen prediction keys locally so the
   static gate can audit them. This is an inexpensive preflight rule, not an extra GPU check.
3. Old canonical publishers must remain auditable after later dossier revisions land. The v13
   publisher and test were corrected to recognize an already-present v13 inside a later v14 record,
   preventing a recurring stateful-test failure.
4. Near-threshold results remain mixed evidence. MLP8--10 was not called sufficient merely because
   one cell reached `0.709`; the preregistered claim required every cell and both CE and margin.
5. Native MLP boundaries are localization handles, not the final semantic basis. The next layerwise
   screen is only a route to identify causal writes that should subsequently be split within MLPs
   and grouped across modules by common downstream use.

## Gates

`CIRCUIT_FOCUS: PASS.` The hour added task-level capability/null results, a new selective circuit,
three exact Task14 causal localizations, canonical Task14 v13 and v14 records, and a tested reusable
writer-decomposition path.

`CEREMONY_BUDGET: PASS, WITH FAST ENGINEERING REPAIRS.` Basic GPU analyses remained seconds long.
The numerical failures stopped scientific interpretation and were repaired structurally without
changing rows, arms, metrics, or thresholds. The additional work was directly necessary for an
honest causal instrument rather than a large suite of backup guarantees.

`NOVELTY_LESSON_GATE: PASS.` The canonical Task14 dossier, registry, active claims, failure records,
and similarly named downstream MLP13--17 experiments were checked before the upstream MLP0--10
split. Invalids and nulls were preserved separately. The recurring lesson remains explicit: high
activation reconstruction is not the scientific target; held-out CE, answer margins, exact
interchange, selective effects, and composition are.

## Immediate continuation

Canonical publication of the depth-group screen is next. In parallel, the next experiment will
preregister a conditional per-layer decomposition inside the empirically motivated MLP4--10 path:
each layer's donor write will be measured both alone and removed from the full MLP4--10 donor set.
That distinguishes an independently useful write from a write whose effect depends on the other
layers. It will retain exact residual bookkeeping, task margin and full-vocabulary CE, same-number
lexical controls, and strict no-op/closure checks. Only a causally supported subset will then be
split within its MLP weights or grouped across modules by common downstream computation.
