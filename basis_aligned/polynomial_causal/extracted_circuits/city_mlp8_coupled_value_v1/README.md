# Coupled direct-plus-MLP8 value operator

`runtime=execute.prepare(torch.load("program.pt",weights_only=True))`, then
`execute.execute(runtime,z,delta,token_ids)` returns the combined direct
attention8 plus MLP8-mediated head9.8 current-value correction.

The edited RMS9 is generated internally from native factors. Native `z8`, the
upstream intervention `delta`, token IDs, and the later model suffix remain
external. The direct and MLP8 terms are folded into one coupled operator; this
package makes no independent-composition claim.
