# Rung 563 correction: natural increment prompts

R562 v1 passed its mechanical CPU checks but used bracketed family labels in the prompts. Those labels could reveal
the intended rule or experimental family to the model. No model output was opened.

R563 freezes the same 160 semantic groups, split assignments, numeric states, seven families, and single-token answer
endpoints after removing every family-revealing label. Base and donor prompts within a target family now have exactly
the same natural prompt form; only the registered numeric state or digit/word representation changes. Reusing a base
prompt across multiple derived rows in one semantic group is deliberate. Oriented prompt pairs must remain globally
unique, and the receipt reports endpoint reuse instead of hiding it with artificial text.

All requirements and limitations in the R562 preregistration otherwise remain unchanged.
