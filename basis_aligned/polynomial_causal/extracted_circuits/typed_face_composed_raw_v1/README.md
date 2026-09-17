# Complete conditional head8/head9 write

Load `program.pt` with PyTorch `weights_only=True`, construct `Program(p['head8'], p['head9'], p['lambda90'])` from `execute.py`, and call:

```python
program.execute(current8, donor_city8, raw_mixed9,
                recipient_token, donor_token, city, destination, strength=.5)
```

The result is the head9 reflection-odd value-write delta. The head8 stage retains both donor key factors and inherited value, keeping recipient current value; the head9 stage holds routing native and changes its current-value branch only. Both stages retain native normalization and rotary conventions. No model or repository import is required when this folder is on the module search path.

Three native arrays remain: recipient normalized block8 attention input `[B,T,1152]`, donor normalized city state `[B,1152]`, and raw mixed block9 input `[B,T,1152]`. Tokens, city and destination mask remain explicit. At T32, this is74,880state floats. Native generation of these arrays and the suffix after head9 are outside the package. The optional earlier MLP7 donor generator is not bundled here and must not be counted as free.

Storage:1,788,419floating scalars including lambda90,20token indices,7,224,941bytes. Native120forward replay and isolated CPU40fixture replay pass every registered implementation gate. Unknown city IDs are rejected; strength zero gives zero write. This verifies mathematical implementation composition, not small causal interactions or a complete circuit. The path's line-break minimum attenuation and random-partition composition failures remain.

[Registration](../../TYPED_FACE_COMPOSED_RAW_V1_PREREGISTRATION.md), [native result](../../TYPED_FACE_COMPOSED_RAW_V1_RESULT.json), [isolated result](../../TYPED_FACE_COMPOSED_RAW_V1_STANDALONE_RESULT.json), [current Logan report](../../explanations/for_logan/research_update_2026-09-17_2215_regional_response.md).
