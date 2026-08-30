# Induction equality tensor FINAL/OOD v2 — implementation retry 1

This is an implementation-only retry of the frozen v2 protocol.  The first
authorized execution terminated before scoring any row because
`model_state_sha256` attempted `scalar_bfloat16.view(torch.uint8)`, which PyTorch
rejects for a zero-dimensional tensor.  The preserved terminal failure is
`induction_equality_tensor_final_ood_v2_failure.json`.

Retry 1 changes no row, arm, component, tensor program, metric, bootstrap,
threshold, or decision rule.  It only flattens each contiguous state tensor to
one dimension before viewing its raw bytes.  This is byte preserving:

\[
\operatorname{bytes}(x)
= \operatorname{bytes}(\operatorname{reshape}(x,(-1))) .
\]

The retry has a distinct authority, outcome namespace, and lock.  It reuses the
already frozen v2 role tensors and requires a new independent source-closed GO
audit before authority can be created.  The original v2 audit, authority, and
terminal-failure bytes are part of the retry source closure.  Every other original
v2 outcome path must remain absent before authority, during collection, and at
terminal replay.  The spent v2 namespace therefore cannot be removed, rewritten,
or reused without invalidating the retry.
