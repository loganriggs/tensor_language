# Regional typed-face conditional executor

`native.py` and `program.pt` run independently with PyTorch. `execute(program,
current, donor_city_state, recipient_token, donor_token, city, destination)`
returns the head8.2 write for the complete routing/inherited face. Load weights
with `torch.load("program.pt", map_location="cpu", weights_only=True)`.

The recipient query stays fixed, both QK factors use the donor city key, current
values stay recipient, and inherited values come from a frozen eight-token table
computed through the native first-layer value map and normalizers. Unknown city
tokens raise. Two native state tensors remain; this is not a token-only model.
Head9 reentry, odd-value propagation, and suffix are external.

Eight captured native examples pass an isolated CPU replay with maximum relative
write error 8.26e-7. The original stricter all-readout recursive replay fails on
three small controls and remains failed. A separately registered fresh midpoint
screen passes sixteen equal-norm null comparisons and four unrelated readout
limits on four context documents. Independent composition fails. No complete
circuit or whole-model compression certification.

Price: 885,761 float scalars plus eight integer token indices; serialized weights
3,546,313 bytes. At32tokens the two native arrays have38,016scalars. Report these
inputs and all external computations alongside the package count. Fewer input
arrays alone does not prove less memory or a simpler complete model.

`execute.py` preserves the earlier factor-boundary algebra prototype for audit.
Its three-term state expansion and behavioral corner identity are exact, but
summing separately propagated state atoms through the suffix is not licensed.

See [manifest](manifest.json), [isolated check](../../check_typed_face_standalone_v1.py),
[full replay](../../ODD_ATTENTION8H2_TYPED_FACE_NATIVE_V1_RESULT.json),
[fresh screen](../../ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_RESULT.json), and
[composition failure](../../ODD_ATTENTION8H2_TYPED_FACE_INTERACTION_V1_RESULT.json).
