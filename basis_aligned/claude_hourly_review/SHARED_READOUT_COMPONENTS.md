# Shared readout components across three auxiliary decisions (bilin18)

Cross-line summary of the definition-of-done batteries run on 2026-09-17 (Claude lane). Every number is an
edit on rows fresh to the line unless marked; receipts are listed in the three scorecards in this directory.

| line | decision | readout set (weight-only `O_h^T(u_a − u_b)`) | fresh-row margin removed | cue reader | how the cue reaches the readout |
|---|---|---|---|---|---|
| aspectual | has / had after since/by | {8.1, 9.1, 9.4} | 50–66% | **8.1** reads since/by (token-only value; 2 constants) | direct (8.1 at the final query) + relay 8.1@bank → 9.1/9.4 |
| temporal | will / had after tomorrow/earlier | {11.3, 9.1, 15.5, 9.4} | 74–83% | **8.1** reads tomorrow/earlier at the subject NP (token-only; 8 constants) | 8.1@NP + MLP8–10 → NP state → 11.3; 9.1/9.4/15.5 read cue + NP |
| narrative | was / is after "Last winter … stood" / "Every winter … stands" | {15.5, 11.3, 9.4, 9.1} | 72–74% | none at head grain (8.1 absent from the top six) | tense already in the second sentence's state; 11.3 and 9.1 read the tail |

What is shared: the block-9 pair {9.1, 9.4} relays in all three lines; 11.3 and 15.5 are the readout for both
tense/temporal lines; head 8.1 is one temporal-cue token reader whose block-0 value branch is a lookup on the
cue token, serving two lines with two tables. What is not shared: the readout contrast (a weight object per
line) and, on the narrative line, the cue path (no single cue token).

Common failure pattern: the strict four-way additivity bar (25% of the smallest single) fails on both four-head
sets by 3–4% of the joint while all pairwise terms are below gate; the largest pair term is a serial relay term
(9.1 also writes the state 11.3 reads). Common open ports: the MLP-generated parts of the relay states (diffuse
writer-pair folds, kill criterion tripped on both lines where tested).

Update 00:03 UTC (v48): on the temporal line, removing 8.1's NP write removes about 32% of the MLP8–10
part of the state 11.3 reads (the MLPs reinforce 8.1's write on that direction); the rest of the MLP part is not
attributable to any few writers at this grain. Per the review-4 kill criterion (≥ 40% required) the MLP relay ports
stay declared on both lines; no further folding of them at head/MLP grain.

## Readout families across nine decisions (reuse census, 00:15–00:20 UTC; opened authored rows, weight-only sweeps)

| decision | contrast | top-4 heads (logits) | set fraction | was−were gate |
|---|---|---|---|---|
| aspectual has/had | has−had | 9.1, 8.1, 9.4, then 15.5, 11.3 | 50–66% (fresh) | passes |
| temporal will/had | will−had | 11.3, 9.1, 15.5, 9.4 | 74–83% (fresh) | passes |
| narrative was/is | was−is | 15.5, 11.3, 9.4, 9.1 | 72–75% (fresh) | passes |
| modal would/will | would−will | 9.4, 11.3, 9.1, 15.5 | 59% | passes |
| perfect have/has | have−has | 11.3 1.54, 7.8, 5.3, 9.7 | 67% | fails (related reader) |
| lexical number were/was | were−was | 11.3 1.67, 5.7 0.60, 7.8 0.32, 9.7 0.24 | 75% | fails (related reader) |
| quantifier number was/were | was−were | 11.3 0.72, 7.8 0.57, 5.7 0.12, 13.1 0.09 | 50% | fails (related reader) |
| coordination were/was | were−was | 5.7 0.47, 11.3 0.42, 7.8 0.33, 9.7 0.09 | 49% | fails (related reader) |
| preposition on/of (non-auxiliary) | on−of | 6.3 0.49, 13.8 0.36, 7.8 0.34, 8.8 0.27 | 28% | passes |

Two readout families at the auxiliary slot: a **temporal family** {9.1, 9.4, 15.5} + 11.3 (tense, mood, aspect;
8.1 as the token-only cue reader where a single cue token exists) and a **number family** {5.7, 7.8, 9.7} + 11.3
(agreement decisions). Head 11.3 belongs to both: the general auxiliary-slot readout. The non-auxiliary contrast
(on/of) uses neither family ({6.3, 13.8, 7.8, 8.8}; 7.8 recurs as a generic helper). For number lines the
was−were reader is related, so the gate that fails there is not a selectivity failure; a number-appropriate
unrelated reader (e.g. has−had) is needed for a proper battery on that family.
