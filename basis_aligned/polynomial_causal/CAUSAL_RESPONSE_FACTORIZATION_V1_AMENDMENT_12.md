# Causal-response factorization v1 — Amendment 12: training grid lifecycle and third price view

Status: prospective and controlling before any candidate fit. The lawful 229-document
training snapshot exists, but no structured candidate has yet been fit. This amendment
does not authorize validation, EVAL, rank selection, or a circuit claim.

## Dense-control repair

Amendment 7 proved outcome-blindly that the original two-coordinate price makes every
strictly matched dense SVD control rank zero: one dense observation vector costs 4,802
persistent values, while every frozen structured candidate costs at most 3,200.

We retain that rank-zero result exactly. We also introduce the already audited
**amortized-total price** as a third, separately labelled, noncontrolling view:

$$
T=P+229C,
$$

where $P$ is persistent values and $C$ is per-document values. A dense rank $r$ costs

$$
T_{\mathrm{dense}}(r)=r(4802+229).
$$

The largest $r$ satisfying $T_{\mathrm{dense}}(r)\leq T$ is reported, but it cannot
replace or relax the registered $(P,C)$ Pareto order. The per-cell training mean is
also retained and explicitly priced at 4,802 persistent values and zero per-document
values. Thus neither control is mislabelled as matched under the original order.

## Immutable resumable training grid

The production entrypoint has no caller-selected input, output, ranks, seeds, device,
fitter, or role. Before opening the training snapshot it must acquire one stable
non-symlink lock inode, verify an exact published source closure, verify a canonical
independent GO audit of those bytes, and reject an unexpected output census. It then
loads only the independently authorized training snapshot and runs the 17
frozen rank pairs at the three frozen seeds with CUDA float32 Adam for 2,000 steps at
learning rate 0.03. Validation and EVAL are absent.

Each rank/seed cell is staged, semantically replayed in full, then published atomically
and create-only with a directory fsync. No fallible verification follows visibility.
A result contains the
canonical CPU-float64 factors, document codes, exact canonical replay loss, health,
literal prices, multiply-add count, phase/owner errors, and worst owner-pair normalized
RMSE. Only the optimizer's two registered nonfinite outcomes become scientific failure
cells; integrity, I/O, protocol, and CUDA-resource errors abort nonzero. Process interruption is
not converted into a scientific failure and already completed cells are replayed, not
refit. A directory lock forbids concurrent owners.

The seeded initialization loss is defined and replayed on CPU float64 from the exact
device-independent random preimage; it is not trusted from a receipt or recomputed by
device-dependent CUDA reduction. Health is derived from this baseline and canonical
final replay. Resumed failures must have exact `RuntimeError` type and one of the two
registered nonfinite messages. Every stored CP factor column is checked for unit norm,
positive maximal-magnitude pivot, and canonical hash order, so a prediction-preserving
scale/sign/permutation regauge is rejected rather than accepted as a second program.

Only after all 51 cells have result-or-failure terminals is a manifest staged and fully
replayed, the exact preterminal census checked, and the manifest linked receipt-last.
It binds the exact source closure, training artifact and tensor identities,
complete cell hashes, training response RMS, strict rank-zero control, per-cell mean,
and false validation/EVAL flags. Source closure is replayed again after fitting and the
terminal directory census must be exact. An existing terminal is replayed read-only and
can never trigger filling a missing cell.

For later validation, every receipt freezes calibration arm budgets
$m\in\{2,4,8,16\}$, the literal calibration-cell count $n=49m$, and the conservative
normal-equation multiply-add proxy

$$
nK(K+1)+K^3,
$$

separately from the per-document prediction cost $4802K$. Training uses zero
calibration cells; these counts authorize no validation access.

The synthetic acceptance gate is: successful multi-cell fit, byte-identical resume,
preserved planted failure, exact census, and no validation/EVAL fields or capability.
Passing this gate validates the runner, not the factorization hypothesis.
