# Fresh conditional error-content confirmation

The discovery screen repaired17/21 native errors with two named content paths.
Test that fixed rule on a complete fresh cohort; the proposed explanation may
fail because errors have different source content, multiple competing wrong
reads, or nonselective paired effects. This is conditional explanation using
known native errors and gold labels, not a deployable detector of model errors.

Use random-layout generator seeds30909/30910, first8 worlds per population:
IID24-cycle and OODthree8-cycle functions, all24 query entities and hops0..3,
both original and graph-preserving renamed inputs. Thus16 independent worlds,
1536 pairs and3072 native states. No trained output opened before registration.
OOD denotes the cycle topology; renamed peers are dependent controls, not extra
independent worlds. Save all native query outputs and all errors without filtering.

For every native wrong state, choose the initial binding with largest total
final-reader wrong-minus-gold logit contribution, using all four heads. Ties use
the first record. Nominate raw if its direct value equals the wrong answer;
otherwise nominate join if its two-step endpoint equals that answer and the
middle record precedes the nominated record. Otherwise mark unsupported and
leave the nominated arm native. Include unsupported cases in error denominators.
Do not select another source or field after seeing removal outcomes.

For each error and its paired context use the same record ordinal and field,
with each context's own states and labels. Arms: native, whole binding-read cut,
nominated content cut, opposite content cut, joint raw+join cut, identity.
Raw=E/8 at the selected value; join=existing L2H1 embedding-endpoint contribution
to both selected record positions. Content cuts subtract the projected path
from final V under native P/RMS/Q/K/residual. Native oracle constructs its own
states and uses projection/pattern hooks. Unsupported cases retain all available
raw/join diagnostic effects but have an explicitly native nominated arm.

Freeze these gates:

- A mechanical: controls pass; every3072 native state represented; every native
  error retained; all-output native/export/path closure max<=1e-9 and relative
  RMS<=1e-10 with denominator floor1e-6; identity and whole-query read-cut closure;
  fixed-gain raw+join composition max<=1e-9; finite outputs and restored hooks.
- B coverage: at least3 native errors per population and3 supported errors per
  raw/join field overall, otherwise insufficient evidence. At least80% of errors
  in each population have a supported nominated field. Report every uncovered
  case and its graph content, with no expansion of the rule.
- C mediation: separately for raw and join over supported primary errors, mean
  wrong-minus-gold margin reduction>=50% of whole-cut mean reduction (positive
  denominator), mean wrongP loss>=.25, and nominated cut repairs>=50% of cases
  repaired by the whole cut (nonempty denominator required). Additionally it
  repairs>=50% of ALL native errors separately in each population, including
  unsupported errors. Report per-population/field cells without hiding empty ones.
- D specificity: at least5 correctly answered paired controls; pooled mean
  absolute goldP change<=.10 and at least95% stay correct. Report individual
  worst damage, per-population/field cells and opposite-path presence; an absent
  joined opposite path is not a live specificity control.

Empty required evidence is untested, not a pass; numerical A is separate from
the scientific gates. Save every named arm's query distribution and scalar
metrics. Report full-vector nominated-versus-whole effects descriptively; they
are different interventions, with no equality claim. No post-failure source,
head, dose, rank, producer expansion or redefinition of the OOD cohort.

Use existing shared extraction/path/scoring code and managed bqrunner, FP64,
screen batches8,1800s and256MiB per new tensor. All387968 constants, native
routing computation, derived dictionaries and context caches remain charged.
No structural saving or complete-model adoption is claimed by this confirmation.
Passing supports a reusable error-content explanation within this task family;
full distribution prediction by a simpler token-to-logit program remains required.
