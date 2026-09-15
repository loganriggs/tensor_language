# Setting2 regional head9.8 QK1 late-group fresh routing V2 correction

V1 stopped before any scientific output while reconstructing the first native
attention call. Its rotary helper squeezed the singleton head axis from the
cached cosine and sine tensors even when rotating a full tensor with nine heads.
That changed the broadcast shape from `[1,T,1,64]` to `[1,T,64]` and raised a
dimension mismatch.

V2 branches only on tensor rank: it retains `[1,T,1,64]` for native four-dimensional
multi-head tensors and uses `[1,T,64]` for the three-dimensional late/remainder
group tensors. The correction changes no arithmetic after broadcasting, rows,
arms, source group, outcome access, prices, metrics, predictions, thresholds, or
interpretation from
`SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FRESH_ROUTING_V1_PREREGISTRATION.md`.
V1 is implementation-invalid and cannot count as evidence.
