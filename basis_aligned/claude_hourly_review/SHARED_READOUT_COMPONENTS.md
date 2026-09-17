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
