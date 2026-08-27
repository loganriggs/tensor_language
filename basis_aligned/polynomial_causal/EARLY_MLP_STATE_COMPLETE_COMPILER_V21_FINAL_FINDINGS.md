# Compiler-v2.1 final findings: real conditional signal, insufficient interface

## Authoritative outcome

Compiler-v2.1 completed its one-shot sealed final evaluation with all eleven
integrity subgates true. The terminal authority is
`authoritative_negative_v21_final`: the package is not admitted and its claim scope
is `negative`.

The authority chain binds the attempt `e20c2b94...a7c2838`, result
`c73f2a7f...9bfb5cd`, manifest `21c1a371...9605dc`, program receipt
`c9c67bdd...8ec2c`, program bundle `36a8e520...c0b614e`, and terminal authority
`659051ed...`. The final role and evaluation callback were each loaded exactly once.
Baseline replay was bit-identical, gauge canaries and original-call poison passed,
the hook was restored, the component tree was unchanged, and every recorded hash and
byte count recomputes. The negative is scientific, not procedural.

## Main numbers

The registered ratios are remaining teacher KL, so lower is better:

| interface | remaining ratio | 95% CI | recovered |
|---|---:|---:|---:|
| MLP0, with exact projected MLP1 | 0.37329 | [0.34733, 0.40327] | 62.67% |
| MLP1, conditional on compiled MLP0 | 0.56451 | [0.52680, 0.60983] | 43.55% |
| compiled MLP0+MLP1 jointly | 0.66308 | [0.62719, 0.70849] | 33.69% |

Admission required every remaining ratio to be at most 0.50. MLP0 passes;
conditional MLP1 and the joint package fail.

In common CE currency on these final rows,

$$
\operatorname{CE}(NNN)=3.84450,
\quad
\operatorname{CE}(OON)=3.61792,
\quad
\operatorname{CE}(QQN)=3.78536.
$$

Thus the exact rank-64 projected pair gains $0.22658$ CE, while the executable pair
gains $0.05914$, or $26.10\%$ of that oracle gain. The registered half-oracle margin
is $-0.05415$, with 95% CI $[-0.06740,-0.04189]$, so the second primary gate fails
decisively.

## What worked

The package is not a null:

- compiled MLP0 alone gains $0.01837$ CE, CI $[0.01155,0.02486]$;
- compiled MLP1 after compiled MLP0 adds $0.04077$, CI
  $[0.02419,0.05649]$;
- the pair beats both singletons with a positive full interval;
- the true-label pipeline beats mean and shuffle controls;
- label alignment passes;
- copy CE improves rather than worsens;
- every registered token-frequency collateral cell is inside its bound;
- relative benefit remains positive on the exact-MLP2 background.

MLP1 alone is harmful on the deployed parent, but helpful after compiled MLP0. This
is direct evidence of a parent-conditional ordered computation, not independently
interchangeable modules.

The MLP2-background result must be read narrowly. Exact MLP2 worsens both the compiled
and baseline arms on these states; the compiled pair merely retains a positive
advantage there. It is not evidence that exact MLP2 improves or compensates for this
compiled pair.

## What the family bank rules out

The failure is not mainly ordinary validation-selector regret:

| same family at both sites | final CE | oracle CE-gain fraction |
|---|---:|---:|
| affine z-only A | 3.81164 | 14.50% |
| state-complete affine Euclidean B, selected | 3.78536 | 26.10% |
| state-complete affine causal C, final best | 3.78279 | 27.23% |
| native Euclidean D | 3.94660 | -45.06% |
| native causal E | 3.86210 | -7.77% |

C improves on selected B by only $0.00257$ CE and remains far below the oracle-half
gate. The expensive native-product grammars are worse. More ridge values, native-K
values, or isolated MLP0 clustering are therefore low-information repetitions.

## Global accounting

No global ledger moves:

- replacement inventory remains `36/36`, which is scope rather than explanation;
- named behavior remains `32.1% +/- 6.4%`, leaving `67.9%`;
- named causal headroom remains `0.57968 / 5.30682 = 10.92%`, leaving
  `4.72714` nats or `89.08%`;
- certified executable recovery of the current ship remains zero against clean CE
  `2.9455`, ship CE `3.8431`, and the `+0.8976` nat/token gap;
- the separate 36-site constant-stake ceiling remains `55.038%`, leaving
  `2.50365 / 5.56837 = 44.962%` in that currency.

The compiler ratios, its $0.05914$ local CE gain, and the 36-site ceiling have
different interventions and denominators and may not be added.

## Pruned strategic order

1. **Same-basis suffix-objective plus explicit physical cross-map discriminator.**
   This directly tests whether the fixed 64-dimensional interface failed because it
   optimized local coefficient error rather than suffix behavior, or because MLP1
   lacks an explicit transported parent-code term.
2. **Gauge-invariant causal residual rank curve and joint suffix-Fisher basis, only if
   priority 1 fails.** This separates inadequate predictor grammar from inadequate
   projectors before paying for a larger latent compiler.
3. **One-support current-ship macro cube.** Mint and decompose the actual $+0.8976$
   gap; only admitted executable programs receive credit.
4. **Typed attention compiler.** Use a low-rank/content-dependent grammar for routing
   and a richer suffix/covariance-weighted grammar for values.
5. **Conditional MLP2, then OOD/edit certification.** Compile MLP2 only after upstream
   transport is adequate, then require second-corpus/code OOD and selective edits.

Blind higher-K/native-product sweeps, semantic naming of arbitrary rank-64 axes,
another MLP2 alignment factorial, and any attempt to promote the rejected package are
pruned.

Priority 1 has begun prospectively in
`EARLY_MLP_SUFFIX_TRANSPORT_V1_PREREGISTRATION.md`. Its pure physical cross-map and
orthogonal-gauge/intervention contract passes all nine independent tests. Mathematical
and lifecycle reviewers approved the protocol for implementation; numerical execution
remains forbidden until the complete runner/source closure is committed, pushed, and
re-audited. No new fit, validation, intervention, or final row has yet been loaded.
