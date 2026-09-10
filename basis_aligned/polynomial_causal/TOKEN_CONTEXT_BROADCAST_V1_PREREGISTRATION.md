# Shared first-value stream as a token-context source

10 September 2026. Original handoff/pilot and both unembedding paths.
Prior art: channels.md, v173/v174 first-value cue/slice interventions, v185
source-versus-live-feedback, correlative first/local branch sufficiency failures.
This tests current lexical context and actual token readers, not discovery of
static V0 or a repeat of function-word correlative source selection.

Current A1/A2/G16 base prompts; cyclic next row is natural lexical donor.
CPU row audit: only token position3 changes, equal lengths9/12/10. Keep the
recipient k_perp reader from TOKEN_CONTEXT_SOURCE_V1_ARTIFACT.pt fixed.

Separate two input ports: ordinary token input (including attention0 output,
initial residual and every embedding re-entry) and the first-value cache
returned AFTER attention0. This cache is reused by attention1–17; alter it
once, preserving the attention0 output. Later local values/routing/MLPs recompute.
Its known exact producer is V0*RMS((lambda0_0+lambda0_1)*RMS(embedding(token))),
evaluated with native operation order (two scalar products then addition).

Per panel: native base, native cyclic donor, base with donor first-values,
donor with base first-values, base with own first-values.5 bodies,15forwards
240seq overall, <=900sec. No fitting, new rank or selected head slices.
No-op output and all native state/readout bridges use maxabs<=1e-3 and rel<=1e-5;
static first-value producer bridge uses same bars. Saved base u17 replay uses
maxabs<=1e-5. All native grammatical base/donor margins must be positive.

For gate vectors d=tau_donor-tau_base, p=tau_patch-tau_base, use aggregate
transfer=<p,d>/||d||^2 and error=||p-d||/||d||. Gate denominators>1e-4 in norm.
Remaining-stream gate magnitude=||tau_donor_with_base_values-tau_base||/||d||.
Also report centered full-vocabulary effects, factorial interaction, and CE.

Predictions:
A instrument: exact15/240 and shared executor controls, all48 base and48 rotated
grammatical margins positive, above bridges, exact single-token cyclic changes,
finite outputs and nontrivial gate norm. Original base gate replay preserved.
B shared context source: A and value-only gate transfer>=.80, gate error<=.20,
and remaining-stream gate magnitude<=.20, separately on BOTH A1/A2.
C lexical behavior: A and ALL16 cyclic pairs on EACH target frame prefer their
own primed bare verb over the other's verb at BOTH native endpoints. Each
recipient-versus-donor lexical margin denominator must exceed1e-6. Without
filtering rows, mean value-only donor recovery>=.80 in BOTH frames and G
base-value-swap meanabsolute correct-token CE change<=.10. If native capability
fails, preserve it and do not interpret a gate result as lexical-circuit evidence.

Report all arms and failed endpoints. B or C failure does not select another
head slice/direction. Factorial interaction is a measured contextual coupling,
not a claim of independent extracted circuit composition. All native parameters
remain; only the known token-to-V0 primitive is independently specified. Shared
first-value content still has contextual routing consumers.

Save gates, per-row errors/recoveries, CE and factorial interaction for paired
CPU continuation. No extra model calls for that audit. A positive source screen
would need held-out constructs, an explicit smaller consumer program, selective
removal and joint intervention tests before promotion.
