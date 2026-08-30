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
or role. It loads only the independently authorized training snapshot and runs the 17
frozen rank pairs at the three frozen seeds with CUDA float32 Adam for 2,000 steps at
learning rate 0.03. Validation and EVAL are absent.

Each rank/seed cell is published atomically and create-only. A result contains the
canonical CPU-float64 factors, document codes, exact canonical replay loss, health,
literal prices, multiply-add count, phase/owner errors, and worst owner-pair normalized
RMSE. A numerical failure becomes a create-only failure cell. Process interruption is
not converted into a scientific failure and already completed cells are replayed, not
refit. A directory lock forbids concurrent owners.

Only after all 51 cells have result-or-failure terminals is a create-only manifest
published. It binds the exact source closure, training artifact and tensor identities,
complete cell hashes, training response RMS, strict rank-zero control, per-cell mean,
and false validation/EVAL flags. Source closure is replayed again after fitting and the
terminal directory census must be exact.

The synthetic acceptance gate is: successful multi-cell fit, byte-identical resume,
preserved planted failure, exact census, and no validation/EVAL fields or capability.
Passing this gate validates the runner, not the factorization hypothesis.
