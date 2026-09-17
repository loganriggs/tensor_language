# Conditional head9 odd-value delta with fewer inputs

Load `program.pt` using PyTorch `weights_only=True`; instantiate `Program(weights)` from `execute.py`, then call `program.execute(raw_mixed9, delta8, lambda90, destination)`.

The raw mixed block9 residual and upstream head8 delta are `[B,T,1152]`; destination is a source-position mask. Both QK factors, their reflection-odd product, native RMS epsilon and rounded rotary factors remain explicit. Inherited-first values cancel because they are identical in the two value arms. Raw mixed state combines the old residual and initial-state reentry inputs. It is still an externally generated native state.

The package stores 901,121 floating scalars, 3,673,871 serialized bytes. It derives its key adapters from these weights. The reentry scalar is explicit. One native-state array plus one upstream-write array remain; including the head8 generator would require current8, donor-city state (or earlier g7), and raw9—three native arrays over that wider boundary. The native suffix remains outside.

All native replay gates pass on40opened sequences; all40isolated CPU fixtures pass. Native max write error1.626096e-6; isolated max2.137687e-6. Max readout discrepancy4.291534e-6logits; per-effect errors3.2312e-5–3.6870e-4 against the registered.001 gate. These checks do not reverse earlier stricter failures or certify fresh behavior of the whole path.

[Registration](../../ODD_VALUE_DELTA_RAW_V1_PREREGISTRATION.md), [native result](../../ODD_VALUE_DELTA_RAW_V1_RESULT.json), [isolated result](../../ODD_VALUE_DELTA_RAW_V1_STANDALONE_RESULT.json), [current report](../../explanations/for_logan/research_update_2026-09-17_2215_regional_response.md).
