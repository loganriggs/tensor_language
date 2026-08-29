# Gauge transport triangle v3 receipt-only recovery

**Frozen before recovery:** 2026-08-29 08:38 UTC

V2 completed all scientific computation and created its result and state, but receipt
publication failed because JSON serialization changes integer dictionary keys to
strings and the publisher compared the deserialized object to the pre-serialization
Python object. V2's mutually exclusive failure binds the complete partial artifacts:

- result SHA256: `2b79648c9866dfbad51c57c6b8536870962db8998040476f9c636fac5994891b`;
- state SHA256: `a85d7cefcc1ea2623dfd0ba42289dbe11b4eb7aec94f923a5492e16a9a069c2e`;
- v2 failure file SHA256:
  `4d6c77993af2a4362345b8654ac5ba5d6640f7be0b04cd2d6e17f5b1c081de7b`;
- v2 authority file SHA256:
  `5f5785e47dea61db6633c6d65d228946f71905e0baf4b9fa7cc0188377410898`.

V3 is receipt-only. It may not load the checkpoint, run the model, alter the result or
state, change a metric, or reinterpret a threshold. It must verify all four hashes,
the v2 failure's self-hash and partial hashes, the result's checkpoint/row/authority
bindings, the complete frozen decision set, and the state tensor keys, shapes,
finiteness, and config equality. JSON semantics are compared after canonical JSON
normalization. If all checks pass, it writes one fresh create-only receipt last.

The scientific outcome remains the v2 result: the full oracle passes; projected-basis
sufficiency, direct transport, and chain composition fail. V3 only converts a complete
receipt-less terminal into a receipt-backed terminal.
