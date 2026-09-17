# Head8.2 write with an earlier donor input

`execute.execute(program, current, donor_g7, recipient_token, donor_token, city, destination)` returns the conditional destination write. Load `program.pt` with PyTorch `weights_only=True`. Import `execute` with this directory on the module search path.

Recipient input is the normalized block8 attention input, shape `[B,T,1152]`. Donor input is the city-position residual after attention7 and before MLP7, shape `[B,1152]`. Token IDs select the native normalized initial-embedding and inherited-value tables. The program computes MLP7 and both RMS stages exactly, retaining both head8 key factors. Unknown city tokens are rejected.

There are still two native activation inputs, or 38,016 floats at T=32. Storage is 16,836,739 float scalars and 40 token indices; serialized program 67,352,243 bytes. MLP7 is now inside the boundary rather than supplied through its output. This is an explicit port relocation, not fewer ports or whole-model savings. Head9.8's conditional odd-value interface and the entire downstream suffix are external.

Native replay on 16 opened sequences passes state/write and registered readout gates. Isolated CPU replay with only this directory and PyTorch passes. Original stricter replay failures remain in historical receipts; this run used its separately preregistered .001 per-effect gate. Prediction, selectivity and composition are not newly certified by implementation equivalence. The source approximation's fresh evidence and both composition failures are recorded in the current Logan report.

[Registration](../../TYPED_FACE_MLP7_DONOR_V1_PREREGISTRATION.md), [native replay](../../TYPED_FACE_MLP7_DONOR_V1_RESULT.json), [isolated CPU replay](../../TYPED_FACE_MLP7_DONOR_V1_STANDALONE_RESULT.json), [Logan report](../../explanations/for_logan/research_update_2026-09-17_2140_regional_source_transfer.md).
