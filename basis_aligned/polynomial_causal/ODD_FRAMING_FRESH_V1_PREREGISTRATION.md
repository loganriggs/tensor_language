# Fresh template and city validation of the O framing relay

Freeze `ODD_FRAMING_FRESH_V1_ROWS.json`:48 rows from two newly authored
templates, Cambridge/Phoenix and Leeds/Chicago, and six frozen spelling pairs.
Rows were built without model execution or scores. Every paired prefix has equal
length, exactly one changed one-token city, no indefinite article before a city,
and one opening quote token defining framing versus copied clause.

Five arms are native; remove O from framing sources; remove O from copied-clause
sources; remove all O; and remove the full S+R+O head9.8 decomposition. This is
240 body forwards with a180-second cap.

- `pred_a`: live framing+clause+pre+city+self source reconstruction of O is at
  most `1e-10`; all values are finite and exactly240 forwards execute.
- `pred_b`: native paired cue contrast is positive for at least10/12 pairs in
  each template, and all-O cue-effect norm is at least5% of full-head cue-effect
  norm in each template.
- `pred_c`: framing-only cue error versus all-O is at most35%, while clause-only
  error is at least50%, in each template.
- `pred_d`: framing control-effect RMS is at most50% of framing target-cue RMS
  in each template.

Failure of B makes source-role transfer inconclusive rather than licensing row
replacement. This is held-out within the controlled regional task, not corpus
OOD. Native upstream states, complete weights and suffix remain. No static
compression, independent input generation, unique semantic unit or quantization
claim follows from a pass.

