# Residual unfolding certificate — Amendment 1: exact target-owner rectangles

Status: prospective after the original production analysis failed closed on incomplete
FIT support, and before inspecting any response or residual singular value. The only
newly inspected object was the Boolean validity mask.

The original preregistration required every one of the 2 × 49 × 49 × 229 cells to be
valid. That condition failed: 70.52% are valid and no document has all 49 targets.
No receipt was created and no zero imputation or spectrum was computed.

The validity mask is exactly broadcast over phase and source and varies only over
target and document. We therefore replace the impossible global rectangle with six
fixed target-owner rectangles. For target owner $h$, retain all of its targets $T_h$
and exactly the documents

$$
D_h=\{d:\text{ every }t\in T_h\text{ is valid for }d\}.
$$

The observed rectangle sizes, determined from the outcome-free Boolean mask, are:

| target owner | targets | complete documents |
|---|---:|---:|
| a8 | 16 | 20 |
| a16 | 13 | 10 |
| m16 | 6 | 46 |
| a3 | 5 | 54 |
| m14 | 5 | 76 |
| m13 | 4 | 109 |

For every source owner $g$ and target owner $h$, the certified tensor is now

$$
E_{g,h}=E[:,S_g,T_h,D_h].
$$

There is no imputation. All six source owners are compared on identical target and
document support for a fixed $h$. The primary registered fork uses target owner `m16`:
`m16 -> m16` is compared with the other five source owners into the same six targets
and same 46 documents. The original 1.5 energy/rank-support and 1.25 weighting
thresholds are unchanged. All 36 raw and residual pair certificates are reported.

This amendment is a missing-data repair, not an outcome-shaped rank choice. It does
not authorize combining rectangles into a pooled SVD, matrix completion, validation,
EVAL, a candidate topology, or any ledger claim. The original failed precondition is
preserved in the later mathematical review.
