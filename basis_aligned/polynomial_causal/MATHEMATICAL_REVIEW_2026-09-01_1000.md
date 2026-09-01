# THREE-HOURLY MATHEMATICAL REVIEW — 2026-09-01 10:00 UTC
Context: the rotation closed at 09:25 (ledger complete through §2478). The parked question is what mathematics opens the NEXT phase. Sign convention §2135: damage = CE ADDED ABOVE NATIVE, LOWER IS BETTER.

## Top three moves (ranked), one executed

### 1. EXECUTED — Joint Tucker-core price/bound map for MLP0 (Eckart–Young in the induced metric)
**Object:** the proposed next representation f̂(x)=U·C[(AQx)⊙(BQx)]+b with input rank r, product width k, output rank p; literal price 1152r + 2kr + pk + 1152p + 1152 per layer.
**Theorem used:** for any core with output rank p, the metric-weighted function error is lower-bounded by the omitted invariant energy of the output-mode Gram beyond rank p (Eckart–Young in the metric where the Gram was measured — rung 381 supplies these numbers exactly).
**Executed enumeration (CPU, receipts only):** 79 grid cores are cheaper than the adopted p768 input-only program (13,272,192 scalars for MLP0). Cheapest per output rank, with the measured bound:
- p=768: 3,540,096 scalars (saves 9.73M more than adopted!) but omits ≥7.5% invariant output energy
- p=512: 2,950,272 — omits ≥17.4%
- p=256: 2,360,448 — omits ≥34.3%
**THE FINDING: price is abundant; damage is the only constraint.** A single Tucker core at one layer could out-save the ENTIRE adopted {4,0} program fourfold — but the adopted program keeps the output map NATIVE (omits zero output energy), and the cores' measured lower bounds start at 7.5% omitted. Everything hinges on one unknown: the conversion gain from omitted invariant output energy to CE. 
**Operational consequence (the 379 pattern):** ONE calibration build — ALS-fit a single corner (recommend r=768, k=2304, p=768; price 7,079,040, still 6.19M under adopted) with census + ray projection — converts the whole bound family into CE predictions and either licenses a region of the (r,k,p) grid or closes the family on price, exactly as the r96 value point closed all value ranks. This is the mathematically forced first rung of the next phase, and its bars can be frozen from the tail-law/ray machinery before any GPU is spent.
**Assumption that may fail:** the induced-metric Gram from 381 uses position-zero exact folding + context covariance; live multi-position behavior may weight output directions differently — the calibration build tests exactly this.

### 2. Flattening-spectrum border-rank certificates (closure instruments)
**Object:** the product mode (width k) of the same tensor. The sym-inner-product identity used in 381 computes the k-mode Gram without materializing 1152³; its spectral tail lower-bounds the product width any CP/Tucker core needs in that metric — a computable stand-in for border-rank lower bounds from algebraic complexity.
**Consequence:** if the k-mode spectrum is flat (like the output mode), small-k cores are certified impossible BEFORE any fit; if it decays fast, the k=1152–2304 corners in move 1's grid are the right targets. **Falsifier cost:** one CPU screen, same machinery as 381. Natural companion to the calibration build.

### 3. Multi-intervention causal-abstraction certificate (bisimulation upgrade)
**Object:** the gate family. Today's signed gate uses one intervention (a16 mean-KO). Causal-abstraction/bisimulation math says a compiled program is a valid abstraction iff an intervention BASIS commutes with compilation. Upgrade: freeze a small named set (a16, m16, one early attn head, one vocab-row family), require signed-effect transport on ALL, report the worst. **Consequence:** turns "intervention-faithful" from an existence claim into a quantified abstraction radius; also directly serves the manipulability goal. Cost: ~140s per intervention per artifact — schedule with the next adoption, not before.

## Pruned
- MDL refinements (358 complete), further covariance reweightings (350/353), finite-state/Hankel (closed ×4), gauge block search (closed ×2), value ranks (379), local rank grids (357/375).

## Recommendation to the next session/phase
Open with move 1's calibration build under frozen bars derived from: tail-law census band, ray-projected cert floor (≥43 for frontier claims), and move 2's k-mode screen run first (CPU) to fix k. The grid, prices, and bounds in this review are frozen paper — the physical rung inherits them.
