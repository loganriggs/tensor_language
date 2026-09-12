# Shared cubic source continuation: objective-scale repair

Continue both terminal native V1 source dictionaries, rank16, query8/source7;
recompile private query writers at source0 without fitting source features there.
Preserve exact coefficient capture and native reference head scales. No data fit.
The only optimizer change is the V2 angle-based sufficient-descent safeguard.
Both objectives use the fixed divisor 0.1809357634075061, the maximum previous
final capture. This positive rescaling preserves minimizers, but changes gradient
stopping units. Original failed V1 predictions remain failed.

A: dense controls and four planted V2 starts pass; native capture finite difference
relative error <=1e-4 and gradient evaluation <=2seconds. No fits if A fails.
B: both arms stationary at projected gradient <=1e-6 in the new common units;
neither capture regresses by more than1e-8. Also report original-unit and
current-capture-relative gradients; these are diagnostics, not replacement bars.
C: >=4 cross-start source matches at absolute coefficient cosine>=.9; each arm
has>=4 atoms used by>=2heads with>=10%of its summed energy per head; summed
individual-atom energy / projected energy<=2. Same sharing definition as V1.

Each arm <=4000updates/300seconds;800second hard limit; zero body forwards.
Reuse frozen V1 coefficient reference estimates and their standard errors;
only capture numerators change. Store source atoms, histories and correspondences.
Readers cost110592floats; native private writers, normalizers and opaque weights
remain charged. Null: this bounded rank16 continuation does not establish stable
shared source features. Local stationarity is not global recovery; structural
sharing does not establish OOD prediction, extraction, selective removal or reuse.
