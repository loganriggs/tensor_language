# Exact donor input moved before MLP7

Extraction test, not a fresh prediction test: use the 16 now-opened native
sequences from KEY_SOURCE_FRESH_V1. Keep recipient normalized block8 state as
one supplied array; replace donor normalized block8 city state by donor g7,
the city state after attention7 and before MLP7. Token IDs select normalized
initial embeddings compiled from native weights (20 tokens, no fit).

Executable owns Left7, Right7, Down7, bias7, lambda8, the token table and the
existing native head8 program. Both RMS stages, full MLP7 product, both key
factors and inherited-value generator stay present. No approximation, truncation
or normalizer omission. Port count stays TWO arrays; donor boundary is earlier.
Charge all added weights; no compression claim from moving weights inside.

Three arms: native, original head8 face, generated donor face. 48 forwards,
300-second budget. pred_a: every donor state relative error <=1e-5 and write
relative error <=1e-5. pred_b: all ten readout absolute differences <=1e-5 AND
relative Frobenius <=1e-6. pred_c: target and each unrelated readout effect error
<=1e-3, denominator norm >=1e-6. These are implementation gates on opened rows,
not new selectivity or OOD evidence. Null: the earlier generator changes the
native write/effect beyond these tolerances. No gate changes after outcomes.

Export the self-contained package and eight-or-more replay fixtures; run an
isolated CPU subprocess with only torch and the copied package, no model import,
CUDA or repo path. pred_d: generated state and write <=1e-5 relative; unknown
token rejected. Package/input counts, bytes and source hashes reported. A native
pass without isolated replay is incomplete extraction. Existing composition
failures and original stricter all-readout replay failure remain preserved.
