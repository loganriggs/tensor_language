# Million-token natural-input panel, 10 September 21:16 UTC

User asks whether at least one million native input tokens can inform the metric
for unsupervised unembedding–MLP17 factorization. Claim on AGENT_BOARD; prior
MLP17 context-covariance work and current512k capture checked. This extends
document coverage and estimates the input metric; it does not identify circuits.

Use the locally cached Pile-10k Arrow file pinned by SHA256 in the rows receipt.
Seed91162114 permutes documents. Take one513-token GPT2 prefix per document of
length>=513, reject exact-text and prefix duplicates, stop at2048. This favors
long documents and early positions. Near-duplicate documents and historical
corpus overlap remain possible. FineWeb local shard was unavailable; Pile is a
named new corpus panel, not certified model-training-distribution data.

Split1600/224/224 documents before capture. Process512inputtokens each:
1,048,576 total, of which819,200 training. Sample32positions without replacement
per document uniformly from0..511:65,536 inputs total,51,200 training. More
processed tokens and documents do not automatically mean more saved fit states.
Stream the mean and second moment over every training position. These statistics
can precondition later fits; they do not specify their empirical fourth moment.

Predictions: A exact512bodyforwards/2048seq and65536sampledinputs; B exact split
and document/prefix uniqueness; C finite scaledFP16 cache, input rounding<=5e-4
and bias-free native MLP output regeneration relativeL2<=1e-3 on the first128
sampled training inputs. Any failure invalidates use at unchanged thresholds.
No loss or convergence claim is registered for capture alone.

Managed lane1 only,900s alarm. Price512bodyforwards plus one128-state MLP replay,
~164MB disk, per-training-batch1152x1152 second-moment accumulation. Store inputs
only and regenerate bilinear targets later. Mean/second moments: FP32 products,
FP64 accumulation; not an exact-arithmetic identity. Preserve the old dataset.
Output and source hashes bind the new capture; test data remain unopened by
fitting/evaluation. Follow-up is a matched representation comparison on these
inputs, with original coefficient metric and new empirical metric both reported.
