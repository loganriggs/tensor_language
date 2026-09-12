# Extracted regional shared head2 branch, token-input mixed precision

This package executes one weight-selected consumer of the two-child shared source component. It preserves the validated native branch and its child interventions on the registered 48-context panel. It is a local extraction with external context inputs.

Load `program.pt` with `torch.load(..., weights_only=True)` and call `execute_mixed_token_head(query, current, token_ids, rotation, program, child_mask=(1.,1.))` from `execute.py`. Query/current are FP32 tensors of shape [N,1152], token IDs are int64 [N], and rotation is the native relative positional matrix [128,128] in FP32. Move all program fields to the input device **without changing their individual dtypes**. The output is [N,1152]; callers sum contributions over causally allowed source positions. A child mask scales the two children independently; (1,0) and (0,1) isolate them.

The complete 50,304-token lookup generates first-input readings without receiving the first-state vector. Source readers, lookup and dual coefficients are FP64; other fields are FP32. Storage is 863,264 scalars / 4,276,480 tensor bytes. This is larger than the older full-source package because it closes one input dependency with a vocabulary table.

[Native validation](../../MIXED_TOKEN_HEAD2_NATIVE_V1_RESULT.json): all frozen A/B/C bars pass; maximum child error 2.783e-7, maximum full-write error 6.522e-8, maximum removal-effect error 7.460e-6. The inherited result arm named `token_fp32` actually denotes this mixed candidate; field dtypes in that receipt are authoritative. The old FP32 child failure remains a separate result.

Query and current-state generators, relative-position generation, native background, final MLP, normalization and unembedding are external. No standalone token-to-logit extraction or unrestricted OOD/manipulation guarantee is claimed. The manifest verifies exact export identity, not those wider properties.
