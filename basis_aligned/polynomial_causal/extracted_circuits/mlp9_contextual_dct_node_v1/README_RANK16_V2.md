# Rank-16 V2 weights

`weights_rank16_v2.pt` uses the same standalone `execute.py` interface as V1,
but freezes the discovery rank at 16 so that low-energy off-diagonal response
terms are retained. `fixture_rank16_v2.pt` is its isolated replay fixture.

The sole activation port remains native pre-MLP9 residual `z9`. The executor
returns the symmetric 4×4 mixed-Hessian response table and has no model,
suffix, tokenizer, donor, or behavioral-reader dependency.
