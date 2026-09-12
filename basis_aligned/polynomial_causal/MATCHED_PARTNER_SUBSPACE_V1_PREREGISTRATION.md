# Exact self term with optimal partner subspace at unchanged cost

Freeze the continued node0 reader a, unit length in H-whitened producer coordinates.
Its exact star is s Mz, s=a^Tz. Decompose

$$
M=(Ma)a^\top+M_\perp,\qquad M_\perp a=0.
$$

Keep the self interaction (Ma)s^2 exactly. The existing671edge graph keeps670
orthogonal coordinate partners. Replace them with the rank670truncated SVD of
R M_perp, where R^T R=Uc^T Uc. Map output factors back through R^(-1).
Because the error partner is perpendicular to a, its star coefficient error is

$$
\frac12\|R(M_\perp-\widehat M_\perp)\|_F^2.
$$

Thus this is the global conditional optimum for670free partner directions with
the exact self term. The old670coordinate partner choice is feasible. Retain
one parent plus670partner input readers and671output writers: the exact same
14,481,792conditional floats as the old node, before U/background. This is not
a search over ranks or the reader a, and no text is used to find factors.

CPU control passed dominance and exact self preservation. Native plan: A numerical
replay/self/perpendicular conditions<=1e-8 and coefficient error no worse than
the old graph; B all developmental and32context swap cells<=.1/sign>=.9/live>=4;
C all removal errors<=.02; D all relative writes<=.05. Preserve separate panel
verdicts and old failures. Same cache limitations, no exclusions or semantic/OOD
claim. Reuse existing native_partner, exact producer compiler and shared scorer.
Implementation/managed execution is the next bounded block.
