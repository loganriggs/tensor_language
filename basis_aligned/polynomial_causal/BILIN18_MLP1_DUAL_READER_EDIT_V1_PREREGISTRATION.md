# Two-reader constraints on independent writes inside MLP1

The exact full-domain linear-plus-norm input quotient is closed by the native
commutator certificate. Next examine a different object: the physical output
writes required to change one existing task reader while preserving the other.
This targets within-module splitting, selective edits and local composition;
it does not claim a smaller input representation or discover semantic features.

Reuse only BILIN18_MLP1_JOINT_READER_WEIGHT_V1_READERS.json. Stack the two
unchanged four-coordinate reader matrices as C (8x1152). If C has full row
rank, D=C^T(CC^T)^-1 is its minimum-norm right inverse. An output edit D_A z
changes task A coordinates by z and leaves task B coordinates unchanged.
P_A=D_A C_A and P_B=D_B C_B are commuting, disjoint (generally oblique)
projectors; P_A+P_B projects onto the joint reader span. A nonorthogonal split
can be locally exact while requiring large writes and causing suffix damage.

A: source hash, dimensions, finite full row rank; CD=I, P_A P_B=P_B P_A=0,
idempotence, joint sum equals orthogonal joint projector; separate/combined
edits and task cross-talk oracle. All maxabs<=1e-9, relative<=1e-9 where the
reference is nonzero; zero identities use absolute1e-9. No rank truncation,
new task fit, changed reader basis, or new task/model forwards.

B: for BOTH task directions, the worst-case required output-write norm is at
most twice the minimum norm needed when the other reader is unconstrained.
Compute the exact finite-dimensional generalized eigenvalue ratio, not sampled
amplitudes. This is a prospective engineering feasibility bar for a selective
intervention tool, NOT a behavioral or circuit-identification bar. No altered
rank/subspace/budget if it fails. Record full task-subspace principal angles,
C condition number, actual writer/reader scalar counts and original-orthogonal
patch cross-talk, with no claims of semantic equivalence from these metrics.

Controls: oblique two-reader fixture demonstrates cross-talk from separate
orthogonal projectors and exact independent dual writes; joint edits and removals
commute; dependent readers are rejected; near-collinear readers show unavoidable
large write gain. Use CPU only, two threads. If B passes, register one native
selectivity test using existing task rows before execution. If B fails, preserve
that instability and do not promote the dual split as the circuit. All prefix,
suffix, native weights and adapters remain priced; local readout invariance is
not final behavior invariance.
