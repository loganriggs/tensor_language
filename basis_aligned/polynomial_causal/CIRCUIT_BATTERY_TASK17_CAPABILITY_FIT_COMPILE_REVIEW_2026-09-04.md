# Independent review: task-17 capability-only FIT compiler

**Reviewed:** 2026-09-04 04:49 UTC.  
**Target:** commit `5da7c8cea21905f59307ff3b3f3633acf57039bd`.  
**Verdict:** **APPROVE, scoped to the frozen CPU compilation contract.**

This approval licenses the next CPU step: build and independently review a model-facing producer and managed adapter
against this exact contract. It does **not** authorize a model call, GPU use, queue edit, enqueue, result publication,
reader localization, or opening SELECT, TEST, or OOD.

## Immutable target and repository state

I compared the Git blobs at `5da7c8cea` with the files in the current working tree after later unrelated commits.
The compiler, FIT authority, owner test, preregistration, task-17 adapter, integration contract, experiment compiler,
artifact package, and managed-entry module are byte-identical. Their relevant SHA-256 values are:

| Object | SHA-256 |
|---|---|
| capability compiler | `c3e8cca7268ee17280dab15f5a5399592db5fc3c6319b4ff693f5c7b6ab259b3` |
| FIT authority file | `b1d33859f15bee8be04719ec532e84057ac70ef150a06e40ae7583ce70a79d6b` |
| owner test | `a988b71dc40db9b1339ca63a1d442df12b28d73c1c8d31f75bb031d08b2b6344` |
| preregistration | `0fea3731f59c8b9f9b1d1e898f2b4dbca65f706406b69f1b3e429e85bc621a63` |
| task-17 adapter | `cf23dddaf34026e573328bbb40d5a115c13b807ae9faccccca4020a1bb057714` |
| integration contract | `b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e` |
| experiment compiler | `64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c` |
| artifact package | `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc` |
| managed entry | `1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81` |

The compiler contains no Torch, model, checkpoint, queue, outcome, old-battery, R593, or science-entry import. The
preregistration and authority were committed before any model outcome exists.

## Independent authority and semantic reconstruction

I regenerated the complete authority using the frozen adapter, public seed `59317`, and 24 groups per phase. Its
canonical digest is exactly
`16307b8bb9273d56f7c3d09cd629fca78fa1db7f110278e959b6ee301cfb7571`. Filtering the regenerated authority to FIT
produces a list exactly equal, including order and every field, to the captured 96-row FIT artifact. The FIT canonical
record digest is `efb8c9c7a4f66b4e816a232d3b8160c36f39d4cc10bcd47c1cb8a76b817be067`; the file is exactly 82,880 bytes.

I then reconstructed, rather than trusted, all prompt and counterfactual fields in the 96 captured rows:

- each text is exactly `Items: <four comma-separated values>. Item <one-based query>:`;
- each stored token sequence equals GPT-2 tokenization of that text;
- joint prompt-plus-answer tokenization has the prompt as an exact prefix and exactly one answer token;
- stored answers, answer-token IDs, list lengths, and base/donor sequence lengths agree with those computations;
- every group contains one linked A1, A2, P, and C panel derived from the same generated list situation (C adds the
  registered duplicate before forming its own base/donor pair);
- A1 changes only the query; A2 holds the query fixed and swaps exactly the queried and alternate payloads; P holds the
  query fixed and replaces exactly one unqueried payload with a novel value; and C queries two positions containing
  the same target while retaining at least one distinct foil;
- `changed_variable`, `answer_changes`, and `expected_effect` agree with the registered transform semantics; and
- the 24 groups in each phase and the phase payload vocabularies are disjoint across FIT, SELECT, TEST, and OOD in the
  full authority. OOD is
  six items while the other phases are four items.

Thus the captured FIT bytes are not merely hash-consistent; their actual computation is the preregistered lookup

$$
f(v,q)=v_q.
$$

## Calls, metrics, and literal price

Independent compilation reproduces:

- spec SHA `64aea22bbe5896e18e17995f23676099b7ba2efd833d97cb2dc56e0c8eb9ba04`;
- call-manifest SHA `0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`;
- metric-manifest SHA `3efad3188a57628ad35466a0585c2e01ff3ff120642a0301f412900aa987f362`;
- complete compiled-contract SHA `526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9`;
- managed dry-run SHA `c325651204737b63a081e8acdb2b73550b2c8349faacff397ee26920ae9a2623`.

The manifest has exactly eight ordered FIT/native/undirected calls: four 24-row base calls followed by four 24-row
donor calls. Every sequence has length 13. The metric manifest has exactly 24 primitive measurements for each of the
eight `{base, donor} x {A1, A2, P, C}` cells. For each side and row, I reconstructed the target and the nonempty foil
set from the union of that row's base and donor payloads, excluding only that side's target. The target never occurs in
its foil set.

The literal price follows directly from the physical manifest:

$$
8\text{ forwards},\quad 8\times24=192\text{ example evaluations},\quad
0\text{ backwards},\quad0\text{ updates},
$$

and two retained float32 values per evaluation give

$$
192\times2\times4=1{,}536\text{ raw evidence bytes}.
$$

## Phase closure, path binding, and mutation attacks

The typed spec declares only FIT, lists SELECT/TEST/OOD as forbidden, has one authority artifact (`fit_authority`),
and has no outcome artifact. The dry run captures that FIT authority plus protocol/preregistration sources; it does
not call the authority generator. Monkeypatching every task-generation entry to raise still leaves the dry run valid,
so no future rows are generated as a side effect.

The managed capture resolves paths under the declared repository root, rejects absolute paths and `..`, rejects a
symlink escaping the root, reads only a regular file through a no-follow descriptor, checks the file identity before
and after reading, and verifies the declared SHA-256. I directly planted and observed rejection of absolute-path,
parent-traversal, escaping-symlink, byte/hash-mismatch, outcome-in-dry-run, and future-phase-artifact attacks.

I also attacked the compiled and evidence objects together. Re-signing the mutable call summary and metric digest after
changing a row ID, then changing the matching primitive key, still fails against the frozen call-manifest digest.
Changing a target token and re-signing the metric manifest fails against its frozen digest. Price changes, future-split
flags, drop-plus-duplicate evidence, transform or side relabeling, extra localization fields, and nonfinite values all
fail closed. The nonfinite case is refused at the strict-standard-JSON boundary before predicate evaluation, so it
cannot produce a scientific projection.

## Capability stop and projector purity

I separately made each preregistered capability clause fail while the other two passed:

1. base accuracy `76/96 < 0.80`, with every cell `19/24 >= 0.75` and positive mean margin;
2. one cell `17/24 < 0.75`, with side-wide accuracy `89/96` and positive mean margin;
3. base accuracy `77/96 >= 0.80` and every cell at least `19/24`, but a negative base mean margin.

Every case returns `hard_abort`, with every one of the seven declared projection fields null. A planted primitive-type
failure also hard-aborts before the science predicate. Replacing the projector with a function that raises proves the
projector is not called on this path. The real projector passes the framework's static closure/global-state purity
check and its order-permutation recomputation check; its `capability_pass=True` field is reachable only after both
hard-abort predicates pass. The compiled and projected key surfaces contain no reader, writer, component, site, or
selection output.

## Test evidence and remaining boundary

The focused owner plus framework boundary suites pass `97/97` in 4.68 seconds. A broader run adding the legacy battery,
experiment-index, and result-contract suites passes `126/126` in 6.33 seconds. My separate checks added 15 planted
refusals, the three isolated capability-fail cases above, exhaustive reconstruction of all 96 FIT rows, and exact
regeneration of the full four-phase parent authority.

This is not yet a runnable scientific experiment. In particular, the CPU compiler does not itself bind a physical
model's output-array positions to the metric rows or publish an immutable evidence/result/receipt package. Those are
correctly left to the next model-facing producer. That producer must be a separately hash-pinned managed executable,
consume this exact compiled-contract digest, save each exact call request and its two `(24,)` float32 arrays, prove
complete prefix/call-directory coverage, construct primitives without reordering or relabeling, validate the exact
measured price, and preserve the hard-abort terminal. Until that implementation receives a new independent approval,
**there is no enqueue authority**.
