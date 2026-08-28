# MLP0: lexical decomposition, causal code, and compiler runtime

## Three objects that should not share one name

The earlier descriptive decomposition is

$$
m_0(t,c)=\mu+C[g(t)]+\bigl(T[t]-C[g(t)]\bigr)+R(t,c)+\varepsilon(t,c).
$$

Here $T[t]$ is the fit-document mean MLP0 write for token $t$, $C[g(t)]$ is a
lexical-class centroid, and $R(t,c)$ is a continuous predictor of the remaining
context-dependent write. This asks how native MLP0 outputs are organized. It is
useful descriptive evidence, but it is not automatically an executable replacement:
the fitted targets came from native MLP0, and good reconstruction does not prove that
the downstream suffix uses the decomposition sparsely or compositionally.

The rank-64 causal interface is a different object. For a registered output basis
$B_0\in\mathbb{R}^{1152\times64}$,

$$
p_0(z)=m_0(z)B_0.
$$

This continuous coordinate vector selects a small output subspace with unusually
large measured downstream causal effect. It is not a 64-class assignment and its
individual axes have an orthogonal gauge, so they do not yet have invariant semantic
names.

A **compiler** is the executable producer/consumer program. It must compute an
approximation $\widehat p_0(z)$ from quantities available at runtime without calling
native MLP0, physically install the predicted slice, condition MLP1 on the resulting
state, and survive downstream, OOD, intervention, and composition tests. Compiler-v2.1
used a low-rank affine producer,

$$
\widehat p_0(z)=b_0+
\left(\frac{z-\mu_0}{\sigma_0}\right)L_0R_0,
$$

followed by the slice replacement

$$
P_{B_0}[N_0]=N_0+
\bigl(\widehat p_0-N_0B_0\bigr)B_0^\top.
$$

It therefore tested the hypothesis “a compact continuous causal interface is
executable.” It did **not** compile the earlier class-centroid decomposition. The
class hierarchy remains one possible producer grammar, but it was deprioritized
because it did not beat matched-byte continuous maps or pass the causal interface
gate. Compiler-v2.1's joint MLP0/1 package recovered only 33.692% of its registered
teacher-KL stake, so it was rejected rather than promoted.

The current suffix-transport experiment keeps the same 64-dimensional physical
interface but changes the objective: it compares local coefficient fitting with
end-to-end suffix-KL fitting, then tests an explicit transported parent-code term
$p_0A$. This asks whether downstream use, rather than output reconstruction, selects
a simpler and more composable producer/consumer factorization.

## Where the recent time went

Cached FineWeb loading is not the bottleneck. The fit, mask, and evaluation tensors
used by the middle-feature run total about 3 MB and deserialize together in roughly
3--6 ms after Python startup. A measured model import/load takes about 14 s. For
representative jobs, process time outside the serialized experiment timer is only
about 0.6--1.3%, and at most about 4% for compiler-v2.1 stages.

The expensive operations are repeated full-stack fitting/scoring and large solves:

| measured run | runtime | main work |
|---|---:|---|
| middle-feature k-sweep2 | 34.75 min | repeated 36-site compilation, feature-selection sweeps, and up to twelve $5760\times5760$ float64 solves |
| whole-model shortfall bands | 22.55 min | recompile the interleaved 36-site stack for seven interventions |
| front-MLP site decomposition | 18.78 min | repeated whole-stack counterfactual compilation |
| compiler-v2.1 MLP0 | 13.11 min | true and shuffled 108-candidate banks |
| compiler-v2.1 MLP1 | 15.61 min | conditional true and shuffled 108-candidate banks |
| compiler-v2.1 final | 5.10 min | one sealed common-row evaluation |
| corrected exact identity repair | 3.25 min | compile once and run live/exact/replay evaluations |

There were real avoidable losses. Two concurrent k-sweep attempts exited after a
combined 16.35 process-minutes, partly overlapping; earlier whole-model hook failures
cost about 9.45 minutes plus shorter reruns. Compiler artifacts also incurred
multi-minute Git-LFS pushes. These are coordination, validation, and publication
inefficiencies, not FineWeb throughput.

The suffix-transport work consumed mostly implementation and independent-audit wall
time rather than GPU time. That was motivated by earlier false identities and role
leakage, but it has reached diminishing returns. The large-logit byte hashes have now
been replaced, removing roughly 393 MiB of student-side CPU transfer per transaction.
The next disciplined move is to finish the thin observed-model adapter, benchmark the
remaining extra-backward tax, and run a small end-to-end pilot before adding more
lifecycle machinery.

## Correction to the exact middle-MLP control

The first supposed exact arm omitted the architecture's separate `Down_bias`; it
computed `Down(Left(z)*Right(z))` even though the native module computes
`Down(Left(z)*Right(z)) + Down_bias`. Its 68.059% result is therefore a zero-bias
ablation, not an identity.

The corrected same-run control gives CE `5.098802047929132` for both the
bias-inclusive hooks and leaving MLP4--15 live. Pooled and maximum per-row
differences are zero, counts match, and replay is bit-identical. Under the old
descriptive denominator this is 67.5533%. The empirical ridge k-curve used separate
arms and remains valid; only the original identity interpretation was withdrawn.
