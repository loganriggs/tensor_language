# Native preflight for joint shared-reader variable projection

This managed GPU job evaluates the checked weight-only kernel on both saved compatible graph starts. It performs no optimizer steps, model-body forwards or corpus fitting. It follows the matched CPU core solves and the planted gradient/gauge controls.

The fitted variables will be global shared readers, private input spaces and normalized output directions. Each evaluation solves all symmetric interaction cores jointly to a true relative normal residual of at most $10^{-11}$. The objective includes the full folded unembedding and the existing 0.01 whole-group energy penalty. The current graph starts contain 1,239,793/1,236,439 floats and 90/119 int64 incidence entries; output whitener, native U, bias and actual model tail remain common required background.

For each start, measure five uncached evaluations: initial coordinates; initial plus $10^{-3}$ times a unit random direction; central differences around the displaced point with step $10^{-4}$; and replay of the displaced point before emitting the graph. Direction seed3001 and execution seed3002 are fixed.

- A: directional derivative relative error at most $10^{-4}$, emitted graph relative RMS error at most $10^{-8}$, and true inner normal residual at most $10^{-10}$, in both starts. The derivative denominator is the maximum of the absolute finite/analytic values and $10^{-8}$.
- B: median uncached evaluation at most10seconds and peak allocated CUDA memory at most8GiB, per start.
- C: initial normalized penalized objective agrees with the independently completed CPU conditional core optimum to absolute error at most $10^{-8}$.

The null is an invalid or impractical instrument. Passing does not establish native optimization, global recovery or behavioral circuits. The planted independent-start recovery initially failed0/4; coordinate re-encoding recovered1/4 but still missed the2/4 bar. An optimizer must check convergence after re-encoding, not merely accept a small raw packed gradient. Consumer-topology changes tested so far did not rescue the selected planted miss; a same-topology coordinate reset did.

The code and transitive local imports, model checkpoint, graph inputs, CPU reference and control receipts are hashed before managed enqueue. Maximum job alarm600seconds. Results determine the next native optimization controller and budget; no long fit is hidden inside this preflight.
