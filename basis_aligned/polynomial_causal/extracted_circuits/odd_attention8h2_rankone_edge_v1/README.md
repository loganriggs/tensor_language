# Conditional inherited-city head8.2 edge

This package executes the identified rank-one write

`routing8.2(destination, city) * c_proj8_head2(lambda8 * c_v0_head2(delta normalized city embedding))`.

`execute.py` takes recipient/donor normalized city-token embeddings and the
caller's native head8.2 city-routing scalars. It returns the residual write at
every destination. The two factor maps and mixture scalar are in `program.pt`.

On all96 native rows, the factored write agrees with direct per-destination
projection within1.73e-7 relative error, and the token-embedding value generator
replays within2.64e-7. `CONTROL.json` independently replays one context from
each of four templates. At20--24 tokens, transmitting `T+1152` factors instead
of a `T*1152` dense write saves94.91--95.75% of the conditional interface and
94.22--95.05% of projection/scale multiplies. The package stores294,913 static
scalars.

This is a conditional edge. Head8.2 routing, normalized embedding generation,
head9.8 O, and the native suffix remain external. Behavioral evidence and
failed semantic-role hypotheses are recorded in
`../../ODD_ATTENTION8H2_CHAIN_FRESH_V1_RESULT.json` and
`../../ODD_ATTENTION8H2_DESTINATION_ROLE_V1_RESULT.json`.
