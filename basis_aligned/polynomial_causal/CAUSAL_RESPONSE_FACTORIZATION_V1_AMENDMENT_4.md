# Causal-response factorization v1 — amendment 4

Status: controlling and frozen before any FIT response bundle is deserialized. This
amendment preserves and repairs the second independent boundary NO-GO. It does not
authorize factor fitting, validation, EVAL, or a scientific claim.

## Preserved failure

Independent audit of amendment 2 reproduced a time-of-check/time-of-use failure. The
outcome-blind binder first recorded receipt, terminal, authority, bundle, and manifest,
then replayed the historical 21-file source closure and independent audit. A synthetic
bundle mutation during that expensive replay was not observed before the old binding
was returned. All 46 numerical tests passed, demonstrating that green algebraic tests
were insufficient for this lifecycle property.

The audit also found that `FitArtifactBinding.from_parent_binding` copied the claimed
`binding_sha256` without recomputing it, and that the training factory accepted a
publicly constructed artifact dataclass. Owner topology and train-only role exposure
were confirmed repaired.

## Controlling repairs

After every historical-source, audit, authority, manifest, protocol, and receipt
semantic check, the binder must:

1. stable-read the receipt and shared terminal again and require exact record and byte
   equality with their initial reads;
2. stable-read authority, opaque bundle, and manifest again and require their complete
   records to equal both the initial read and terminal receipt aggregate; and
3. recheck failure-terminal and live-owner-lock absence.

The reproduced mutation must now fail with `bundle changed during terminal replay`.
This closes mutations during expensive semantic validation. It does not create an
atomic authority boundary by itself; the future factor-analysis authority freezer must
perform an adjacent final parent replay immediately before its create-only authority
link.

The training factory now accepts the complete parent-binding mapping, not a caller-made
`FitArtifactBinding`. It recomputes the logical SHA-256 over every parent-binding field,
then privately reduces the verified mapping to the immutable artifact identities. A
changed bundle digest under the old binding hash must fail before response exposure.

## Remaining gate

The second audit remains immutable NO-GO because the source-closed analysis authority,
one-use exact-byte `BytesIO` loader, capability poisoning, candidate-result lifecycle,
and receipt/failure boundary do not yet exist. These components must consume an exact
independent GO bound to their complete source closure. Neither this race repair nor the
accelerated optimizer authorizes opening the FIT bundle.
