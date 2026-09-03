# Hourly strategic review — 2026-09-03 23:21 UTC (Claude lane)

Sign convention (§2135): every CE number below is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER.
Frontier (§312/§2125): norm-2304 at 2.6735 added-CE units, unchanged; nothing in this hour installs into it.

## Explained fraction (strict ledger, unchanged this hour)

5.348% / 10.923% / 4.727 nat / 0 of 68 behavior-level circuits. Nothing in §2746–§2756 moves the strict ledger: the width
program is an IMPLEMENTATION description (where each sublayer reads and writes), not a behavioral circuit with selective
removal or reuse. Largest gaps remain: tail dictionaries / coverage credit; the m16 remainder; attn5's write = the price cliff;
and R585 (Codex) as the one live behavior-specific circuit attempt (its 23:17 run exited 1 on an input-shape error — Codex's lane).

## What landed this hour (Claude lane, §2746–§2756, all preregistered, all scored as written)

- §2751 whole-model width program: 36 read frames at k = 768/896/1024 → .197/.096/.034.
- §2752 the readout side-channel is broad (eff rank 218 of 384); truncation is not free.
- §2753 the early read frames are per-site: leave-one-out +.070, neighbours-only +.081 over own .057 (b, c, d FALSE).
- §2754 the settled region begins at block 8, causally: 16 own frames + ONE frame for blocks 8–17 = .0374 at k = 1024 (a–e TRUE).
- §2755 the early frame drift is NOT a low-rank in-span/complement swap (three nulls MET; saturates at +.040 by m = 64–128).
- §2756 blocks 8–17 read AND write through that one 1024-frame, remainder (≈ 7% energy) to the readout: .0362 (a–e TRUE).

State of the width program: ONE 1024-dim bus for the back half (20 sublayers), 16 site-specific 1024-frames for blocks 0–7, a
readout side-channel, total .036 nat added. The early frames resist every compression tried (shared, grouped, windowed,
leave-one-out, low-rank drift). Enqueued: frame_principal_angle_spectrum_probe (construction-free drift dimension).

## Candidates (brainstorm across structure classes)

Tensor / basis: (T1) principal-angle spectrum of the early frames (running); (T2) does the early frame track the TOKEN
EMBEDDING frame — U_s vs top-768 of the wte covariance (block 0–2 frames may just be the embedding's own frame rotating
toward the bus); (T3) the bus frame U_8 vs the unembed's row space — is the bus the readout's frame?
Polynomial / bilinear: (P1) inside the bus, do the late MLPs' Left/Right reads share a smaller sub-frame than their Down writes
(read-k vs write-k asymmetry at fixed total)?
Gauge: (G1) is U_8 unique up to the trivial ≥ 384-dim intersection — replace U_8 by the top-1024 of blocks 11–17 only (§2745's
late core widened) and check the price; if equal, the bus is a property of the late half, not of the averaging.
Causal / behavioral: (C1) selective removal WITHIN the bus: rank the 1024 bus directions by their causal CE contribution per site
(§2116-style top-k selector, label-free) and test whether a 512-direction subset chosen by causal score beats the top-512
eigen-subset (§2118 closed this on the frontier; the bus is a different object, but the closure's logic — metric-constructed
subsets don't beat variance on CE — is a strong prior; low information gain unless the test is cheap: 2 arms).
Program: (S1) the 17-frame program's parameter count vs the model: 17 × 1152 × 1024 ≈ 20M frame parameters + a 1024-wide
back half — state it exactly in the explanation, no GPU.

Pruned: C1 (closure prior §2118, low gain); P1 (needs read/write split machinery inside the MLP — costlier; defer).

## Ranked top five

1. T1 principal-angle spectrum (running; construction-free; falsifies "low-rank drift" independent of §2755's construction).
2. G1 bus uniqueness: U_8 from blocks 8–17 vs widened late-7 core vs blocks 11–17 core — 3 arms, 480 forwards. Tells whether the
   bus is "the late frame" or a compromise; changes how the program is written.
3. T2 embedding frame: are blocks 0–2's frames the wte covariance frame? 2–3 arms; explains what the early rotation starts from.
4. T3 bus vs unembed row space: capture of W_Uᵀ's top-1024 by U_8 and the CE of reading the unembed through U_8 (readout on the bus
   — is the readout side-channel needed only because the unembed reads outside the bus?). 2 arms.
5. S1 exact program statement in the explanation (no GPU; do now).

## Executed

T1 is running (enqueued 23:17). G1 is registered and enqueued next (this review's action). S1 goes into the explanation file
written this hour (explanations/explanation_2026-09-03_23xx.md).
