# Glossary: program jargon → standard technical terms

Written 2026-08-26 per user request. Ledger sections before §1502 use the left
column; writeups from §1502 on default to the right column.

| Program jargon | Standard term |
|---|---|
| stand-in / plank | **module replacement** (approximation of one component's function) |
| ship / glass ship | **joint replacement model** — model with N components simultaneously replaced |
| total glass | joint replacement of ALL components |
| glue | **trained low-rank correction** — rank-32 linear map + bias added to a replacement's output, trained by SGD on full-model cross-entropy |
| anchor / optimal-ablation anchor | **optimal constant baseline** (learned constant replacement per component, Li & Janson 2409.09951; frozen for scoring) |
| fid_opt | **normalized CE recovery**: (CE_const − CE_repl)/(CE_const − CE_clean) |
| stake / Δ_opt | **ablation cost** — CE increase when the component is reduced to its optimal constant |
| board / priority board | component ranking by **unexplained CE** = ablation cost × (1 − best recovery) |
| edge | **inter-module weight path** — composed weight matrix from module A's hidden units into module B's input maps |
| channel | **low-rank interaction subspace** — top singular directions of that composed matrix (whitened by input RMS) |
| mean transport vs signal | the **constant (mean) component** vs the **input-dependent (centered) component** of an inter-module path |
| kernel / distance kernel | **relative-position-averaged attention pattern** (pattern as a function of query−key offset only) |
| roster / specialists | **exempt head set** — heads kept exact because position-averaged patterns fail for them |
| three-tier grammar | the three-level attention approximation: offset-averaged pattern / per-head rank-32 QK truncation / exact head |
| lin2 / linall | **ridge regression** from [attn_L, mlp_{L−1}] / from all upstream module outputs |
| tier table (tier2000 etc.) | **token lookup table**, top-K frequent tokens exact + low-rank tail |
| handle / handle score | **intervention quality metrics** — how well a basis supports targeted ablation (removal), targeted reconstruction (extraction), and out-of-sample prediction (generalization) |
| lookalikes / jailbreak bucket | **surface-similar non-members** (adversarial controls for membership prediction) |
| crew / committee | **head ensemble** jointly implementing one behavior (e.g. the 13-head capitalization ensemble at layers 13–17) |
| ledger | the research log (BILIN18_CONNECTION.md), or an exact additive decomposition of stream contributions |
| driver tick | one iteration of the scheduled orchestration loop |
