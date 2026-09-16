# CrossFirst direct cubic V1 implementation correction

The first execution of runner hash
`f5f686fd4648bd6f56eda64026bf64561efc77aab9199b751806a3987d7a2dd6`
terminated before writing an artifact or result.  On the second triple-JVP of
the first prompt, PyTorch rejected the model's mutable rotary cache because a
cached tensor had escaped a completed nested forward-AD level.  No prediction
was scored and no scientific quantity was observed.

The corrected runner replaces `Rotary.forward` locally, for this process only,
with its cache-miss algebra: construct positions, outer-product them with the
unchanged `inv_freq`, and return bf16 cosine and sine tensors.  This removes
only mutation/reuse of cached tensors; it does not change the formula, model
weights, prompts, tangent vectors, suffix, derivative orderings, predictions,
or thresholds.  The corrected runner receives a new hash and remains bound to
the original preregistration.
