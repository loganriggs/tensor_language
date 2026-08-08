# HANDOFF — the six-architecture comparison slice (GRID.md phase V1)

Written for a reader with **no memory of this work**. Everything below is on
disk; nothing needs a transcript.

Cell: **depth 2, width 128, vocab 8192 trained byte-level BPE**, Muon 0.02
matched across every arm (no lr sweep), AdamW 0.004 on the embedding, 15,000
steps x batch 16, single epoch, identical data order.

## What the slice found (full write-up: RESULTS.md FINDING 11)

All five non-vanilla variants **open the attention-to-attention path the plain
model leaves numerically shut**, and all five **acquire induction at width 128**
where the plain model needs 256. Verdict: DIFFERENT, not a relabelling.

## 1. State of the cells

**DONE — trained AND interpreted** (`{stem}_interp3.json` exists):

| stem | note |
|---|---|
| `tf_vanilla_d2_w128_b8192_s0/s1/s2` | reference; s0/s1/s2 all interpreted |
| `tf_vanilla_d2_w256_b8192_s0` | the decisive control: HAS induction, path still shut |
| `tf_vanilla_d1_w128_b8192_s0` | depth-1 matched null for the natural-text swap probe |
| `tf_{slots,bandwidth,predicate,codebook,shrink}_d2_w128_b8192_s0` | the six-way comparison, seed 0 |
| `tf_{slots,bandwidth,predicate,codebook,shrink}_d2_w128_b8192_s1` | seed-1 replication, all five |
| `tf_slots_d2_w128_b8192_s0_writeinit_only` | mechanism arm: n_slots 1, lasso 0 -> vanilla + nonzero decoder init only |
| `tf_slots_d2_w128_b8192_s0_nolasso` | mechanism arm: partition + per-slot norm, no lasso |
| `tf_{slots,bandwidth,predicate,codebook,shrink}_d1_w128_b8192_s0` | depth-1 matched nulls (trained; some interpreted) |
| `tf_bandwidth_d2_w128_b8192_s0_slot32` | matched-embedding control (trained) |

**STILL TRAINING** when this was written (chains run independently; check
`tf_variant_train2.log` and `tf_variant_train3.log` for `CHAIN2 DONE` /
`CHAIN3 DONE`):

* seed 2 of all five variants at d2 w128 — `slots` and `bandwidth` done,
  `predicate` in flight, `codebook` and `shrink` queued (`tf_variant_chain2.sh`)
* `tf_variant_chain3.sh`: `tf_vanilla_d2_w128_b8192_s0_lr{0.01,0.04}`,
  `tf_slots_d2_w128_b8192_s0_lr{0.01,0.04}`,
  `tf_predicate_d2_w128_b8192_s0_slot32`

## 2. The exact command to finish the job

A waiter is **already armed** (`tf_interp3_master.sh`, launched detached) that
does all of this by itself once the training chains exit. **Check whether it is
still alive before doing anything:**

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
pgrep -f -x "/bin/bash ./tf_interp3_master.sh"      # alive -> just wait
tail -3 tf_interp3_final.log                        # its progress
```

If it is **dead or was reaped**, re-launch it — it is idempotent:

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
setsid nohup ./tf_interp3_master.sh > /dev/null 2>&1 < /dev/null &
```

Or run the same thing by hand (this is exactly what the script does):

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate
# wait for training first:
until grep -q "CHAIN3 DONE" tf_variant_train3.log; do sleep 60; done
# FORCED re-analysis of every slice cell with the CURRENT code (~85 s each,
# ~30 cells, ~45 min).  Forced, not skip-if-present: the analysis code changed
# mid-run (the live-rows content spectrum, the per-head ablation, the induction
# route split), and a table whose rows came from different code revisions is
# the exact failure this slice exists to avoid.
for f in tf_slots_*.pt tf_bandwidth_*.pt tf_predicate_*.pt tf_codebook_*.pt \
         tf_shrink_*.pt tf_vanilla_d2_w128_b8192_s[012].pt \
         tf_vanilla_d2_w256_b8192_s0.pt tf_vanilla_d1_w128_b8192_s0.pt \
         tf_vanilla_d2_w128_b8192_s0_lr*.pt; do
  python tf_interp3.py --stem "${f%.pt}"
done
python tf_variant_compare.py            # writes the table
```

**Where the comparison lands:** `tf_variant_compare.json` (machine readable,
includes `across_seeds`, `vs_vanilla`, `depth1_matched_nulls`, and
`dropped_because_produced_by_an_older_analysis_revision` — that list must be
EMPTY in the final run) and `tf_variant_compare.txt` (the printed table).

**Gate that must be re-run and must pass before the table is believed:**

```bash
python tf_interp3.py --control      # exit 0 = pass; writes tf_interp3_control.json
```

It checks that the variant-agnostic pipeline reproduces `tf_interp2`'s numbers
on the vanilla checkpoint (last run: every ladder stage to 1.8e-6 nats, the
composition budget to 1.0e-6, the stream geometry to 4.0e-6 relative, the OV
composition to exactly 0.0).

## 3. What is left to write up once the table lands

Four things are **not yet in RESULTS.md** because their arms had not finished:

1. **Seed 2** for all five variants — fold into the FINDING 11 tables as
   mean +- sd over three seeds. Seeds 0 and 1 are already in the JSONs and
   already agree (see §4).
2. **The learning-rate arms.** The claim to test: *vanilla does not produce
   induction at Muon 0.01 or 0.04 either.* If it does, the headline becomes a
   learning-rate effect and FINDING 11 must be retracted. This is listed as
   objection C2 in `tf_variant_reviewer_round_1.json` and the claims there are
   explicitly marked provisional until it lands.
3. **The matched-embedding (`_slot32`) arms** for bandwidth and predicate.
   Objection C3. Note this can only affect the C/D/E rows — `slots` already has
   vanilla's *exact* parameter count (1,638,656 total, 590,080 body) and shows
   the full effect, so the count cannot explain the headline.
4. **Depth-1 matched nulls** for the natural-text swap probe, per variant. The
   probe has a large depth-free baseline: vanilla depth 1 scores +0.0670
   (t = 4.49) with a synthetic induction of -0.0276, so the depth-2 numbers must
   be quoted as an EXCESS over the same variant's depth-1 cell.
   `tf_variant_compare.py` computes this automatically as
   `natural_swap_excess_over_depth1_null` once the depth-1 cells are interpreted.

## 4. Claims that rest only on the departing agent's context — RE-DERIVE, do not cite

Everything below was computed and read off, but is **not** in RESULTS.md or any
JSON summary. Each line names the command that regenerates it.

* **Seed-1 replication table** (regenerate: read
  `tf_{v}_d2_w128_b8192_s1_interp3.json` -> `read_ablation_causal.kl_from_model`
  and `rung3_induction`). Observed: layer-0 attention *resampled* out of
  layer-1's read, seeds 0/1 — slots 0.1234/0.1274, bandwidth 0.1504/0.1489,
  predicate 0.0713/0.0741, codebook 0.1080/0.0973, shrink 0.1476/0.1339,
  against vanilla 5.5e-6/4.0e-6. Induction seeds 0/1 — slots +0.1129/+0.1133,
  bandwidth +0.0965/+0.1789, predicate +2.5934/+2.6378, codebook
  +0.0540/+0.0358, shrink +0.0510/+0.1032.
  **Reading: the ROUTING number is tight across seeds; the induction MAGNITUDE
  is not (shrink doubles, bandwidth nearly doubles) though the sign is stable
  and every value clears its power floor.** Report it that way, not as a mean.
* **Predicate seed-1 mechanism replication** (`predicate_induction_split` in
  `tf_predicate_d2_w128_b8192_s1_interp3.json`): zeroing the 16 previous-token
  match scalars leaves +0.0240 (99.1% removed; seed 0 was +0.0330, 98.7%);
  zeroing all named terms lands on -0.0145, i.e. vanilla's null. The rotary
  knockout costs 0.524 (seed 0: 0.532) against vanilla's 3.429.
* **Bag scores are flat across variants** (`rung3_induction.bag_score_mean`):
  0.0846 / 0.0868 / 0.1000 / 0.2289 / 0.0896 / 0.0839 for
  vanilla / slots / bandwidth / predicate / codebook / shrink. So the induction
  difference is **not** a bag-effect difference — the order-free component is
  the same in every arm while the order-specific one moves from -0.014 to
  +0.113. This is a control worth stating explicitly in the write-up.
* **The wiring table's tilt** (`mechanism.read_slot_occupancy`): layer 1's
  queries and keys put 0.40-0.47 of their group mass on slot 0 (where layer-0
  attention writes) against ~0.20 on the MLP's slot, while layer 1's values put
  only 0.11-0.14 there. Caveat that must travel with it: the token remnant is
  full width, so a slot holds *an embedding chunk plus one module's write*, and
  mass on a slot not yet written is reading the embedding.
* **The lasso prunes nothing** at 3e-5: `mean_live_slots_per_read` is 4.00 of 4
  in every slot variant. In RESULTS.md already, but the underlying per-matrix
  shares are only in the JSONs.
* **Not yet checked at all:** whether the *seed-2* cells reproduce the route
  split (`induction_route_split`) and the ladder reordering in predicate
  (attention knockout 2.608 > MLP knockout 2.174 at seed 0, true in no other
  cell in this program). Both are single-seed observations as things stand and
  RESULTS.md says so.

## 5. Two bugs fixed during this slice — do not reintroduce

* `tf_model._qz_slot` hard-cast activations to float32, so
  `cast_model(model, float64)` **raised** rather than ran and the **codebook
  variant had never been through the fp64 tier of the fold gate**. Now quantises
  at the codebook's own dtype; bit-identical in fp32, so no training behaviour
  changed. All twelve variant x depth gates pass (`tf_variant_selftest.out`).
* The MLP content spectrum on a **masked** decoder (slots, shrink) was measured
  over all 128 output rows, but `write_out` discards every row outside the
  module's slot, so 96 of 128 never get a gradient (row norms 100.5 inside slot
  1, 4.7 outside). "Entropy rank 51 against a null of 123" was 32/128 and
  nothing else. `rung2_v` now reports `mode0_unfolding_live` with a
  shape-matched null; **quote that, never `..._ALL_ROWS_DO_NOT_QUOTE`.**

## 6. Files

| file | what |
|---|---|
| `tf_interp3.py` | `VariantFold` + the one variant-agnostic analysis path; `--control` is its gate |
| `tf_variant_compare.py` | builds the comparison table from the `*_interp3.json` files |
| `tf_variant_predictions.json` | predictions P1-P8, registered before the first training step |
| `tf_variant_reviewer_round_1.json` | the mandatory self-red-team, 16 objections with status |
| `tf_variant_preflight.json` | reduction battery + parameter accounting at the cell shape |
| `tf_variant_train_chain.sh` / `_chain2.sh` / `_chain3.sh` | the three training waves |
| `tf_interp3_master.sh` | waits for all waves, force-reanalyses everything, builds the table |
| `RESULTS.md` FINDING 11 | the write-up |
| `MAILBOX.md` 08:25 UTC | the cross-box entry |
