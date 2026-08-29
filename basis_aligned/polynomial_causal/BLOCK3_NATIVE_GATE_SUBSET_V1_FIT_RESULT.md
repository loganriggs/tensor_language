# Block-3 native-gate subset v1 — fit-stage result

## Status

This is a sealed **fit-stage** result.  No validation or final rows were opened, and no
global explanation, storage, causal, extraction, removal, or OOD ledger receives credit.

The short result is:

> Selecting 256 or 512 native product gates carries real signal relative to matched
> random subsets on the joint four-term fit objective.  The fit payload does not by
> itself determine the error of the summed deployed MLP write.

The run is bound to source commit `3c1f8be82cb36f43fa6ec1af055b1b7831e205f1`.
It used 92,160 token positions from 480 receipt-bound fit documents.  Exact RMS replay
had maximum absolute error `9.5367431640625e-07`.  The measured collector ledger made
120 calls to each attention site 0--3, 120 calls to each MLP site 0--2, **zero** calls to
native MLP3, and 240 explicit typed `Down` banks, exactly as registered.

## What was fitted

MLP3 has 4,608 native bilinear product gates.  For the exact polarized inputs

\[
u=\gamma(h+a)h,\qquad v=\gamma(h+a)a,
\]

one gate subset and one decoder were fitted jointly to all four typed writes

\[
uu,\quad uv,\quad vu,\quad vv.
\]

The deployed all-term program forms \(u+v\) and evaluates only \(K\) products.  It is
not charged as a four-bank diagnostic.  All fitted programs are executable float32,
and their direct \(K\)-product output replays the four-term polarization to relative
error below `9.3e-7` in the deterministic runtime check.

## Fit results

The reported fit NRMSE stacks `uu`, `uv`, `vu`, and `vv` as four separate targets.  Its
numerator is the sum of their four squared errors, and its denominator is the sum of
their four native squared magnitudes.  Zero is exact; one is aggregate typed-term error
as large as the aggregate typed-term target.

| Family | Gates | Stacked typed-term fit NRMSE | Literal bytes | Native-byte fraction | Products/token |
|---|---:|---:|---:|---:|---:|
| activation selected | 256 | 0.75659 | 3,545,600 | 5.566% | 256 |
| random within top-1,024 | 256 | 0.79042 | 3,545,600 | 5.566% | 256 |
| permuted-label control | 256 | 1.00412 | 3,545,600 | 5.566% | 256 |
| activation selected | 512 | 0.68491 | 7,086,592 | 11.124% | 512 |
| random within top-1,024 | 512 | 0.71050 | 7,086,592 | 11.124% | 512 |
| permuted-label control | 512 | 1.01371 | 7,086,592 | 11.124% | 512 |
| native MLP3 | 4,608 | 0 by definition | 63,705,600 | 100% | 4,608 |

The activation-selected arm improves NRMSE over matched random by `0.03383` at 256
gates and `0.02559` at 512 gates.  The permuted-label arms near one show that the
regression is not succeeding without aligned writes.  Thus activation-aware selection
is informative, but the useful signal is distributed across many more gates than the
registered budgets retain.

## What this does and does not establish

The preregistered local threshold is **summed all-term write** NRMSE at most `0.20`:

\[
\frac{\lVert
(\widehat W_{uu}+\widehat W_{uv}+\widehat W_{vu}+\widehat W_{vv})
-(W_{uu}+W_{uv}+W_{vu}+W_{vv})
\rVert_2}
{\lVert W_{uu}+W_{uv}+W_{vu}+W_{vv}\rVert_2}.
\]

The sealed fit statistic is instead

\[
\sqrt{
\frac{\sum_q\lVert\widehat W_q-W_q\rVert_2^2}
     {\sum_q\lVert W_q\rVert_2^2}}
.
\]

These quantities are not interchangeable: errors among typed terms can cancel when
summed, and native typed terms can cancel in the denominator.  The payload did not
retain the cross-term moments needed to derive summed-write NRMSE.  Therefore `0.68491`
must **not** be compared to the `0.20` gate and does not yet show that the deployed
all-term interface is inaccurate.  It does suggest that independently editable typed
terms are hard to preserve at these budgets.

Held-out evaluation must compute both the actual summed local write and downstream
consequences directly.  It must distinguish:

1. high local and high final error: this subset grammar fails behaviorally;
2. high local but low candidate **and mirror** final error, with decay across cuts:
   a downstream-null direction, useful for behavioral compression but not a faithful
   Block-3 port;
3. low one-sided candidate error but high mirror error: downstream compensation, not a
   safe null or independently editable component.

The finite-suffix-fitted family is also still absent.  Therefore even a validation
failure for activation selection cannot reject every 256/512 native-subset program; it
would reject this activation-fitted family, while a consequence-fitted candidate
remains the required alternative before closing the grammar.

## Artifact chain

- collector authority: `block3_native_gate_fit_v1_authority.json`
- sufficient statistics: `block3_native_gate_fit_v1_payload.pt`
- collector receipt: `block3_native_gate_fit_v1_receipt.json`
- deterministic-fit authority: `block3_native_gate_subset_v1_fit_authority.json`
- float32 programs: `block3_native_gate_subset_v1_programs.pt`
- fit results: `block3_native_gate_subset_v1_fit_results.json`
- deterministic-fit receipt: `block3_native_gate_subset_v1_fit_receipt.json`

The receipt-bound payload contains sufficient statistics, not the full
`92,160 x 4,608` gate matrix and not teacher logits.
