# C512 and the compressed copy gate compose

Status: exploratory composition on exposed cached documents 33--128.  C512 itself is
a frozen previously evaluated program; this new cross does not upgrade its prior
fresh-data authority.

## Main result

The C512 approximation of MLP0 `Down` preserves the shared 256-dimensional state used
by the compressed L8 copy gate, and the two simplifications compose with a very small
interaction.

All six preregistered gates pass.

## Components

### C512

C512 retains native MLP0 RMSNorm, Left, Right, coordinatewise product features, and
Down bias.  It replaces only the $1152\times4608$ `Down` matrix by a frozen rank-512
continuous program.  Its serialized `Down` program is 5,904,640 bytes versus
21,233,664 bytes native, a 72% `Down` reduction.

### Shared-HOSVD copy gate

The L8 H3/H4 gate reads

$$
z=V_{256}^\top x^{(8)}
$$

from the normalized contextual state entering L8, reuses $z$ across eight Q/K cores,
and writes the shared $\lambda_8v_1$ successor payload.  Its gate plus writer uses
851,968 values versus 1,474,560 native values.

## Factorial

The two letters name MLP0 and the L8 gate:

- `NN`: native MLP0, native gate;
- `CN`: C512 MLP0, native gate;
- `NH`: native MLP0, HOSVD gate;
- `CH`: C512 MLP0, HOSVD gate;
- `ZN`: MLP0 write deleted, native gate; causal scale control.

All other components remain native.  The HOSVD intervention changes only the exact
H3/H4 successor source edge at repeat-eligible positions.

## Output behavior

| Arm | All-scored $\Delta$CE | All KL | Copy $\Delta$CE | Copy top-1 |
|---|---:|---:|---:|---:|
| `NN` native | 0 | 0 | 0 | 88.86% |
| `CN` C512 only | +0.00220 | 0.00462 | **-0.00208** | 88.59% |
| `NH` HOSVD only | -0.00020 | 0.00067 | +0.01054 | 88.52% |
| `CH` composed | **+0.00264** | 0.00470 | **+0.00906** | **88.38%** |
| `ZN` delete MLP0 | +2.59136 | 2.62793 | +2.80118 | 45.92% |

C512 slightly improves copy CE even though its aggregate effect is a small loss.  The
joint program stays well inside the frozen +0.0075 aggregate and +0.03 copy margins.
Its copy top-1 loss is 0.48 percentage point.

The CE composition interaction is

$$
I=\Delta\mathrm{CE}_{CH}-\Delta\mathrm{CE}_{CN}
-\Delta\mathrm{CE}_{NH}.
$$

It is only `+0.00064` nat all-scored and `+0.00061` nat on copy-positive positions.
Thus almost all joint error is explained by the two singleton changes; there is no
large hidden incompatibility at this interface.

## Does C512 preserve the downstream copy state?

On all scored positions, comparing C512 with native $z$:

- $R^2=0.99553$;
- mean position cosine = `0.99846`;
- relative RMS error = `0.05589`;
- fraction of the zero-MLP0-to-native squared-error gap removed = **99.63%**.

On copy-positive positions, $R^2=0.99522$ and zero-gap removal is 99.59%.

The zero-MLP0 arm matters for interpretation.  A high correlation could be trivial
if MLP0 barely affected $z$, but deleting MLP0 causes 2.59 nat aggregate CE damage
and 2.80 nat copy damage.  C512 reconstructs nearly all of its effect on this
downstream-defined state.

## Does HOSVD remain valid on the C512 state?

Yes.  Under native upstream state, H3/H4 scalar $R^2$ is
`0.97817 / 0.96219`.  Under C512 it is `0.97852 / 0.96301`.  The low-rank gate is at
least as accurate after the upstream simplification.

This is stronger than showing that each program is good alone.  C512 changes MLP1
and later trajectories; the HOSVD gate recomputes from that changed live state, and
the composed forward still passes.

## What this tells us about MLP0

MLP0's complete 1152-dimensional write is large and its internal semantics remain
partly gauge-dependent.  But for this important natural-text copy consumer, the
relevant contract is much simpler:

1. C512's rank-512 `Down` approximation constructs almost the same L8 shared state
   $z$;
2. the L8 gate needs only that 256-dimensional state;
3. the resulting conditional lookup remains behaviorally faithful when both are
   installed.

This validates downstream-defined simplicity: C512 is useful here not merely because
it reconstructs MLP0 output or final logits, but because it preserves a specific
causal state and composes with a separately compressed consumer.

## Next step

The telescope should now advance to MLP1 and MLP2, not reopen MLP0 clustering.
Existing evidence says C512's largest internal discrepancy appears in the induced
MLP1 write and is attenuated by deployed MLP2.  The new $z$ contract provides a
sharper assay:

- replace or project MLP1 under C512;
- measure its effect on $z$, the copy edge, and aggregate CE;
- cross the promoted MLP1 arm with deployed/omitted or compressed MLP2;
- retain only programs that compose with both C512 and HOSVD.

Fresh natural-text confirmation is still required before the composed local program
becomes a final claim.

## Artifacts

- `C512_COPY_GATE_COMPOSITION_PREREGISTRATION.md`
- `run_c512_copy_gate_composition.py`
- `c512_copy_gate_composition_results.json`
- `COPY_EDGE_SHARED_HOSVD_FINDINGS.md`

