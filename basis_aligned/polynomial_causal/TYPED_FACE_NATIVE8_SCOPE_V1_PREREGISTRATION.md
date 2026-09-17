# Scope test: conditional odd-value path versus native attention8 application

Keep frozen retained half-write and opened prospective40sequences unchanged.
Current operator installs the head8 write into a virtual block9 residual and
propagates only its changed current value through head9's odd routing. It bypasses
MLP8 and other consumers. This test applies the SAME head8 delta directly at
attention8 output, before MLP8, then recomputes all native later operations.
Neither counterfactual silently replaces the other.

Arms: native,old conditional midpoint,native8 midpoint,16native8 norm-matched
random directions =19*40=760forwards/300s. Same output-subspace directions,
per-destination norm matching, paired signs and four unrelated readers. Seeds
17092300+1000*null_index+context_id, as previously frozen. Same destination mask.

pred_a: native/conditional scores replay prior prospective native/midpoint within
1e-5maxabs; zero writes outside destinations; null norm discrepancy<=1e-5;
finite and760forwards. pred_b: native capability>=18/24pairs and both conditional
and native8 targetRMS>=1e-5 per family. pred_c: conditional midpoint predicts
native8 midpoint effect within .35relativeL2 in every family (sufficiency test).
pred_d: native8 attenuation positive>=.75 of capable pairs,mean>=.02 everyfamily.
pred_e: native8 targetRMS>=2median16nulls and beats>=15/16 everyfamily.
pred_f: every native8 unrelated RMS<=.5targetRMS everyfamily.

Opposing outcome: other consumers amplify, cancel or redirect the same write,
so conditional selectivity or effect prediction does not carry to native8.
Any failure restricts the claim; do not alter support,strength,rows or gates.
No partition-specific composition or fresh-row confirmation is included.
Future suffix changes require the forward response census of this native8 edit;
the old conditional census cannot supply its response attribution.
