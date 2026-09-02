# Rung512 preflight addendum: archival weight precision and installed-write verification

**Frozen:** 2026-09-02 23:58 UTC, after the first managed no-outcome CUDA smoke failed and before any rung512
scientific outcome.

The first smoke retained no task, circuit, relation, or semantic outcome. All28 branch removals, all consumer hooks,
and both directions of the test consumer patch were live. Two instrument checks failed.

First, the source built the archived MLP11 question form from the already-loaded bfloat16 execution model. The frozen
checkpoint intentionally stores the MLP11 `Left`, `Right`, `Down`, and unembedding weights in float32, but constructing
the deployed model with `dtype=bfloat16` downcasts them. The resulting eigenvalues were`+144.82809/-73.83215`, while
the archived float32-weight form is`+144.8641/-73.8464`. The registered object is the archived form, so the repair
memory-maps the same hash-pinned checkpoint and constructs only this fixed basis from its original stored tensors.
The model execution and every intervention remain bfloat16. The registered`1e-3` archival eigenvalue check is not
relaxed.

Second, the patch verifier compared the requested float32 tensor with the installed bfloat16 tensor after converting
the latter back to float32. A maximum difference of4.0 was ordinary bfloat16 rounding, not a failed patch. The repair
compares the capture with `requested.to(installed_dtype)` exactly. The consumer patch itself is unchanged.

No branch, consumer, relation, document, mask, threshold, fitted scale, prediction, route, or execution price changes.
The same no-outcome smoke must pass before the scientific script can be enqueued.
