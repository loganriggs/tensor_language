# Hourly strategic review — 2026-08-29 11:00 UTC

## Bottom line

The best next experiment is still the eight-candidate E4 attention-copy screen. This
hour materially shortened its critical path, but did **not** produce an E4 model
result. The physical intervention and native trajectory are implemented; the current
work is making the fit-only input/mean transaction honest and replayable before using
the GPU.

An independent audit caught that an engineering unit test deserialized the real
combined fit-row container before a fit authority existed. No E4 model, logits, loss,
candidate effect, final role, or OOD role was opened, but the container also held copy
labels. We preserved this as an erratum rather than claiming it did not happen. The
repair is a separate input-only projection containing exactly 192 by 256 input token
IDs and 192 ordered document IDs. The fit runner will never load the combined
label-bearing container.

## How much of the model is actually explained?

These currencies answer different questions and must not be merged into one headline.

| Currency | Current value | Plain meaning |
|---|---:|---|
| Structural inventory | 36/36 sites | Every attention and MLP site has some measured surrogate or intervention. This is coverage, not understanding. |
| Removal-certified storage | 29,196,288 / 545,904,054 = 5.3482453% | This fraction of stored values has a tested removal/compression claim under the strict ledger. |
| Named causal cross-entropy | 0.57968 / 5.30682 = 10.9233025% | Named mechanisms account for this fraction of the measured cross-entropy gap. Cross-entropy is the average negative log-probability assigned to the true next token. |
| Unnamed cross-entropy | 4.72714 nat | Most predictive behavior still lacks a named, causally tested mechanism. |
| Terminal practical actions | 0/68 | No registered final extraction/removal/OOD action has yet passed the strict end-to-end standard. |

Thus the honest whole-model answer is: all sites have been touched structurally, but
only about 5.35% of storage and 10.92% of the registered causal CE currency have strict
credit. The model is not close to fully reverse engineered.

## What concretely advanced this hour?

### E4 physical computation

For a selected set of attention heads, the intervention is

\[
w_{\mathrm{arm}}=(w_{\mathrm{native}}-w_{\mathrm{selected}})+\mu_{\mathrm{fit}}(p).
\]

Here (w_{\mathrm{native}}) is the full native attention write,
(w_{\mathrm{selected}}) is the physical write of the selected head or heads, and
(\mu_{\mathrm{fit}}(p)) is that write's mean at sequence position (p), estimated
only on the fit documents. Parentheses matter in bfloat16 arithmetic. The eight
candidates are six individual heads, the registered four-head path, and the registered
late pair.

The collector now has one intended production path:

1. validate a source-closed, fit-only authority;
2. acquire an inode-and-nonce-owned lock;
3. load only the sanitized 192 by 256 input artifact and derive its ordered-document
   digest;
4. hash the checkpoint immediately before and after loading it;
5. run one native 18-layer trajectory per batch, with every attention and MLP native;
6. independently decompose only the six licensed heads at layers 5, 7, 8, 13, and 14;
7. accumulate one CPU float64 addition per document, then publish both the float64
   master and float32 runtime means;
8. semantically reload the bank with file hash-before/load/hash-after; and
9. recheck the lock, source commit blobs, parent receipts, protected inputs, output
   hashes, and tensor semantics immediately before writing the sole success receipt.

The bank/result publisher is no longer callable with an arbitrary caller-created mean.
It requires an opaque capability minted only after the owned native collector closes.
The current broader E4 CPU suite passes 83 tests. This is implementation evidence,
not behavioral evidence.

### Whole-vocabulary reliability results from the parallel lane

S1918 found that the cached token-only confidence signal also orders agreement on
uncovered current tokens, although less strongly than on covered tokens. S1919 then
raised the uncovered fallback map from rank 64 to rank 512. Reliability improved by
about 0.63--0.78 gradient units across roles, but this closed only 9.7--13.9% of the
gap to covered-token reliability. Therefore rank 64 is one bottleneck, not the main
bottleneck. A larger linear fallback alone is unlikely to solve uncovered-token OOD
failure.

## Largest remaining gaps

1. **E4 fit provenance and publication.** The sanitized input projection and fit
   lifecycle are undergoing adversarial audit. No projection receipt, fit authority,
   or fit mean exists yet.
2. **E4 selection scorer.** After fit means exist, logits must be reduced immediately
   into document-level CE, KL, accuracy, and specificity statistics; raw logits must
   not escape. This scorer still needs the same owned lifecycle treatment.
3. **Unexplained residual CE.** The 4.72714-nat unnamed remainder dominates every
   claimed explanation.
4. **Uncovered-token function.** Rank-512 improves the current linear fallback only
   modestly. Its missing structure is likely nonlinear, conditional, or interaction-
   dependent.
5. **Composition and compensation.** Independent local replacements can be accurate
   yet fail together because later MLPs compensate. We still need an exact whole-model
   composition account rather than a list of local fits.
6. **Practical editability.** No extracted circuit has yet passed selective removal,
   limited collateral CE, and ID/OOD replication together.

## Ranked next actions

### 1. Finish the E4 fit transaction and run the eight-candidate copy screen

Highest expected information gain and causal relevance. It tests a short physical
path near the output, has frozen falsification gates, composes on the live sequential
trajectory, and costs one moderate fit pass plus nine selection forwards per batch.
If no candidate passes, copy localization is a clean negative. If one passes, it gives
the first concrete target for extraction/removal and a 64-mask finite-state assay.

### 2. Correct native-Down consequence Gram

For the MLP0-to-downstream interface, use Jacobian-vector products to find reachable
output directions, orthogonalize them, then vector-Jacobian products to score which
physical product columns are observable downstream. A consequence Gram is a matrix of
downstream response inner products; it ranks components by what later computation can
actually detect, not by local weight energy. This is causally relevant and falsifiable,
but its numeric protocol and implementation remain behind E4.

### 3. Run a small 37-arm exact hybrid telescope

Construct nested models that replace one of the 36 sites at a time. The final-output
difference telescopes exactly into 36 sequential increments even through RMSNorm and
other nonlinearities. Measuring scalar CE and maximum logit change on a small pilot
will reveal whether local errors cancel or accumulate. CPU implementation is cheap;
GPU cost is moderate and can stop early if the bound is vacuous.

### 4. Build a risk-controlled cached-token compiler

The token-only margin is precomputable for essentially the whole vocabulary and
empirically orders agreement. Use document-cluster confidence bounds to accept cached
predictions only where risk is certified. This may yield executable cost savings and
OOD prediction, but it does not by itself explain or remove a causal mechanism, so it
remains below the physical E4 and composition experiments.

### 5. Fit a native-free nonlinear fallback only on uncovered tokens

S1919 makes another pure rank increase lower return: rank 512 closed only about one
tenth of the reliability gap. The next fallback should test conditional mixtures,
sparse dictionaries, or low-degree tensor features on uncovered tokens, with covered
tables frozen. This could improve whole-program coverage but is costlier and less
causally direct than the first four moves.

## Pruned or deferred

- Another generic local-MSE decomposition: it cannot resolve downstream relevance or
  composition.
- Pure fallback rank expansion beyond 512: S1919 gives a measured diminishing-return
  warning.
- Generic CP/HOSVD or gauge norm minimization without a downstream response metric:
  these can change coordinates or weight energy without improving prediction,
  extraction, removal, OOD transport, or certification.
- Tight q64/q128 shared-basis searches: already heavily explored and lower expected
  causal return than E4 or the consequence Gram.
- Final/OOD E4 roles: forbidden until a selection candidate passes.

## Highest-priority safe action executed

The E4 fit critical path was advanced on CPU: the input-exposure failure was preserved,
an input-only projection transaction was implemented, the production fit lifecycle was
rewired to consume only that projection, adjacent checkpoint and artifact hash checks
were added, and adversarial late-write/lock tests were added. The expanded suite passes
83/83. No GPU launch was attempted because the independent audit has not yet granted a
GO and the parallel reliability lane occupied the GPU during most of this work.

Strict ledgers and E4.1--E4.3 therefore remain unchanged.
