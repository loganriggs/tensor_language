# MLP1 implicit folded-tensor v2: authoritative findings

Date: 2026-08-28

Status: post-outcome interpretation of the authoritative, weights-only v2
diagnostic. This document adds no model outcome and authorizes no rows, fitting,
forward pass, CP claim, causal claim, or replacement. It applies the prospective
decision boundary in `MLP1_IMPLICIT_FOLDED_TENSOR_DECISION_ADDENDUM.md`.

## Authority and integrity audit

The publication chain is intact:

- source/weight authority SHA256:
  `baa1eb9bb245c792f9a8ac473d5ca25f0012b55f9aef5b5a8e86f6c5f693ce92`;
- result SHA256:
  `2cbd5a745a6669d49da22ded23bd6df385f3cc057b686f814ad0157ef5fa8281`,
  exactly the 23,297,069 bytes bound by the outcome authority;
- outcome-authority SHA256:
  `b96573ee373e6d4f0032667acb5c838d5205fc9c166eb9e5112d90396c56a9de`;
- protected-snapshot fingerprint:
  `b3556706ad938e99dc200dbc765b7ff0be5af362ebbf52e4f1d65e4901c07fc3`
  in source authority, result, and outcome authority; and
- the outcome authority has status `authoritative_weight_diagnostic_only`, binds
  the exact source authority and result, and records the v2 failure path absent.

The result's internal `complete_pending_v2_last_written_outcome_authority` status and
`none_until_v2_outcome_authority_exists` marker are correct immutable pre-publication
statements, not a conflict: the separately written outcome authority now supplies
authority. The v1 source/failure hashes also replay exactly, while the v1 result and
outcome-authority paths remain absent.

All 19 source-closure blobs replay at source commit
`d414f1bfb1d10d7771a8d0fc9ed05f99c1a58ee4`, which is reachable from
`origin/main`. The source authority binds the prospective decision-addendum hash
`e8f6695e35e8de5746b3ce60b3f0164d8435aa610e117b32174e7c785e6f76c7`.
The result reports zero model forwards, no rows, no materialized folded tensor, and
no raw checkpoint tensor publication. This audit did not access the checkpoint.

The numerical receipt is internally consistent. There are no dead gates; the
maximum balancing log-norm defect falls from `0.4986217010` to
`1.2560739669e-15`. The output and both input unfoldings have numerical rank
1,152. The two input summaries are exactly identical. Output and input Gram traces
are both `134315750.03623924`, with reported relative residual zero. All Gram
eigenvalue minima are positive—output `22563.6581751`, input `432.2739804`—so no
PSD clipping is implicated. Every projected core has exact reported input symmetry.
Recomputation from the serialized spectra reproduced their totals, cumulative
fractions, threshold ranks, and numerical ranks; core/full-tensor ratios and all
native, Down, CP-contract, dense-core, and COO price formulas also replay.

## Authoritative coefficient geometry

The registered energy ranks are:

| squared coefficient-Frobenius energy | output rank | input rank | balanced-Down rank |
|---:|---:|---:|---:|
| 90% | 835 | 937 | 846 |
| 95% | 962 | 1,033 | 970 |
| 99% | 1,103 | 1,123 | 1,105 |
| 99.9% | 1,147 | 1,147 | 1,147 |

Thus MLP1 is full numerical multilinear rank and has no low-dimensional Euclidean
mode-energy knee at any registered threshold. This is stronger information than the
earlier randomized output-rank lower bound, while agreeing with it.

The equal-mode projected cores confirm that the registered small-subspace route is
not close:

| core | fraction of full tensor energy | dense floats | products |
|---:|---:|---:|---:|
| $16^3$ | 0.0002441462 = 0.0244146% | 40,192 | 136 |
| $32^3$ | 0.0008770062 = 0.0877006% | 91,776 | 528 |
| $64^3$ | 0.0033212004 = 0.3321200% | 281,728 | 2,080 |

The top-COO points retaining about 94% of each **core** retain only 0.0229840%,
0.0825740%, and 0.3130040% of the **full tensor**, respectively. High within-core
retention must not be presented as high tensor retention. No sparse curve over a
large/full-rank core was tested.

## Decision under the prospective addendum

### Pruned

1. **Exact dense Tucker compression in coefficient space is pruned.** Exact
   numerical ranks require $(r_o,r_i)=(1152,1152)$. The resulting direct symmetric
   Tucker program would store 767,730,816 floats and execute 664,128 products,
   versus native 15,926,400 floats and 4,608 products.
2. **Every registered 90--99.9% dense coefficient-Frobenius Tucker branch is
   pruned.** Even the marginally necessary input ranks force 439,453; 534,061;
   631,126; and 658,378 symmetric products. Pairing them with the marginally
   necessary output ranks gives lower-bound complete dense prices of 368,985,751;
   516,066,074; 698,697,482; and 757,803,406 floats. Actual joint-core retention can
   require still larger ranks. These lower bounds already exceed native in both
   primary coordinates.
3. **The registered 16/32/64 low-dimensional core-first route is pruned.** Its best
   full dense core retains only 0.33212% of the coefficient tensor. Sparsifying those
   tiny cores cannot repair the omitted 99.66788%.
4. **Balanced Down is pruned as an exact simplification and as a gate reduction.**
   Exact rank 1,152 costs 17,253,504 standalone floats, more than native, and every
   Down rank still executes all 4,608 products. The 90%-energy rank 846 costs
   15,490,944 floats—only a 2.7342% storage reduction—and is licensed at most as a
   weak, storage-only 90%-coefficient baseline. The 95% and higher Down ranks already
   exceed native storage.

These pruning statements are metric- and family-specific. They do not prune a
natural-activation-weighted Tucker basis, a Fisher/suffix-response basis, an
untested large-rank sparse core, or CP.

### Licensed or still open

1. **A changed-metric discriminator is licensed.** The next tensor test may ask
   whether the large Euclidean tails are rarely excited on natural MLP1 inputs or
   weakly read by the suffix. It must use a prospectively frozen activation covariance
   or Fisher/causal-response metric, fit/validation separation, and the same complete
   price ledger. Success there would establish distribution- or consequence-weighted
   compression, not contradict this coefficient-space negative. Failure would show
   that the diffuse coefficient geometry is also physically consequential.
2. **Direct CP remains logically open but receives no positive finding.** For a
   $q$-product approximation, output unfolding rank is at most $q$. Therefore the
   observed output spectra give necessary coefficient-energy lower bounds
   $q\ge835$, $q\ge962$, $q\ge1103$, and $q\ge1147$ at 90%, 95%, 99%, and 99.9%,
   respectively; exact numerical replay still requires $q\ge1152$. These are lower
   bounds only. The result explicitly records
   `cp_fitted: false`, so it supplies no finite CP upper bound below the native
   4,608-gate construction.
3. **Only a separately frozen bounded CP search can advance CP.** Structural prices
   remain favorable for any constructed $1152\le q\le4607$: for example CP-2048 is
   7,079,040 floats, CP-4096 is 14,156,928, and CP-4608 exactly equals native
   15,926,400. But price feasibility is not existence. A valid search must serialize
   the executable factors, report conditioning and cancellation, replay coefficient
   error, then pass natural-state/Fisher and causal controls. Failure of selected
   ranks or optimizer budgets prunes only those registered searches, not all CP.

## Claim boundary

The authoritative conclusion is narrow: MLP1's folded quadratic coefficient tensor
is diffuse across both physical output and input mode subspaces, so the registered
low-rank Euclidean HOSVD/Tucker route is not a useful standalone simplification.

It is invalid to conclude that MLP1 is behaviorally incompressible, that its natural
activation image has the same effective rank, that the suffix reads all Euclidean
directions equally, that CP rank is near 4,608, or that no simpler causal interface
exists. Coefficient Frobenius weights every quadratic coordinate uniformly;
activation covariance and Fisher/suffix sensitivity generally do not. Conversely,
a future activation- or Fisher-weighted success would be conditional simplicity,
not an exact compression of the coefficient tensor measured here.
