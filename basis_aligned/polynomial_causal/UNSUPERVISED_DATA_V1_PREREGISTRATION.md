# Large unlabeled MLP17 data capture

User authority: substantial unsupervised structural search with data, convergence
and red-team review; see explanations/2026-09-10/unsupervised_structure_campaign.md.

Use frozen UNSUPERVISED_DATA_V1_ROWS.pt: 1000 cached 513-token rows, 64 sampled
positions per row from 64..511, fixed random 800/100/100 train/validation/test rows.
No labels determine capture or sample selection. No fitting during this run.
Save actual normalized MLP17 input and native MLP17 output, including Down bias.
Dataset fits must account for that bias; weight-only homogeneous fits keep it separate.

pred_a_capture_instrument: exactly250 forwards1000 sequences512 tokens and
input/output arrays both [1000,64,1152]. Actual first-attention hook counts calls.
pred_b_split_integrity: 1000 distinct whole-row hashes and disjoint exhaustive
800/100/100 row partitions. No document-level independence claim.
pred_c_storage_fidelity: finite FP16 saved arrays; each batch's relative L2 rounding
error <=5e-4 for both inputs and outputs. Maximum absolute native values reported.

Managedlane1 only,900s hard bound. Around295MB local state artifact; compact source
rows and manifest are durable in Git. Never delete unbacked research artifacts.
This is a data/instrument receipt, not a circuit identification or behavioral screen.
