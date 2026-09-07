# Temporal/is-was five-MLP all-position SVD ladder — preregistration

The per-row-mean source basis erased position-conditioned variation. For each of the five frozen MLP
sources, this run instead concatenates every donor-minus-base output delta from token 0 through the
registered semantic position on the old-family training population, and fits its right-singular basis.
Fresh targets and controls remain sealed.

Arms are full patches, ranks 8/16/32/64/128 at every source, and the rank-64 orthogonal complement.
Select the lowest rank satisfying response projection at least `.8`, response RSE at most `.2`, and
behavior projection at least `.75` on both tasks.

Predictions: A atomic authority/split/full replay/finiteness passes; B at least one rank is functional;
C the selected lowest functional rank is selective (median KL at most `.02`, zero flips); D selected
rank is at most 64; E rank-64 complement has absolute response and behavior projection at most `.25`
on both tasks. Maximum 36 forwards; no gradients, objective fitting, or model updates.
