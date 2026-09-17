# Typed routing/inherited face: algebra prototype

Status: **not certified extraction**. This is a factor-boundary executor prototype,
not a new circuit. It takes recipient/donor routing, recipient current values,
recipient/donor inherited values, mixture, output weight, and two position masks.
These are supplied inputs, not closed token generators. Native head9 reentry,
odd-value propagation, and suffix remain external.

For routing A, current value c and inherited value i, retain recipient c:

`delta = [(A1-A0)((1-lambda)c+lambda*i0) + A0*lambda*(i1-i0)
          + (A1-A0)*lambda*(i1-i0)] Wout.T`.

Source and destination masks are explicit. The three writes sum to the difference
of corners 5 and 0. After any identical nonlinear downstream evaluator F, the
complete behavioral face also telescopes to F5-F0. This does **not** mean the
three state atoms may be propagated separately and added after the suffix.

Run `../../check_odd_attention8h2_typed_face_v1.py` from any working directory.
It checks synthetic state algebra and both saved native behavioral panels.
The stronger small-interaction criterion fails on opened replication rows; see
`../../ODD_ATTENTION8H2_TYPED_FACE_INTERACTION_V1_RESULT.json`.
