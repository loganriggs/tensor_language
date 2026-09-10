# Data-fit execution repair V2

V1 data_product_s0 failed before any optimization because the native checkpoint
stores `transformer.h.17.mlp.Down_bias`, not `Down.bias`. The pending V1 data-block
wrapper was cancelled before execution. Original failure/log/source remain.

V2 corrects that key and checks its shape in CPU dry runs. It retains the V1
32 configurations, representations, objective, optimization and convergence bars.
Only data runs use V2, with new result/checkpoint names. Weight V1 jobs continue.
A CPU integration check compares captured training outputs with native bilinear
weights plus this bias. Relative output replay must be <=1e-3; this is a storage/
execution instrument check, not a scientific structural prediction.
