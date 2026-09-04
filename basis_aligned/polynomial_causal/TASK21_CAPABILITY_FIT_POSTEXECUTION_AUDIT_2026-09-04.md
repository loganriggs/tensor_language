# Task 21 capability-FIT post-execution audit

**Audited:** 2026-09-04 UTC. **Verdict: VALID SCIENTIFIC `hard_abort`, not an invalid
instrument.** This is an independent CPU/read-only audit of the immutable task-21 package published in commit
`d59c7f428eeae03778fa6c6c3576d7a8397feb68`. The package exactly implements the preregistered FIT-only native
capability screen, and its frozen native-capability predicate fails. Every registered projection field is therefore
correctly null. This result closes task 21 without localization, resampling, threshold relaxation, or later phases.

No model, checkpoint, GPU, queue, enqueue, runner, service, later-phase authority, or localization artifact was
opened or executed in this audit. The published result, receipt, and evidence were read but not changed. The only
writes are this audit and its append-only board receipts.

## Exact authorization and outcome chain

The outcome commit descends from every approved build/review object below. The task-21 package paths have no diff
against the outcome commit, and the current closure rehashes at all 19 of 19 roles.

| Object | Exact identity |
|---|---|
| authority/compiler build | `9ebab94615eade27b1eb63e4f2c6239337b71dc9` |
| authority/compiler approval | `ca088ce0906160958a2586cff50b707699b7eb88` |
| blocked producer build | `000a113eed35c7e8fac0d2ceed126925963cd0d7` |
| producer approval | `6b8fe576594bb82a5a2093f2338603040739c9af` |
| authorization successor | `fc3f5c16deb7d3bcb035e3def6fcf53bf75ac9c4` |
| outcome package commit | `d59c7f428eeae03778fa6c6c3576d7a8397feb68` |
| authorized adapter | `43564464637c7c0fa7a609ec55bc05377c1d872ad0d0cdf1ef80e957e5026779` |
| authorization amendment | `a31cf24ec79d86f084c29bdc18a909e1ff0457b4e0921fd6249f722adf2b08d1` |
| final pre-execution review | `7972a71fc8759e0d909e73edb08d521c4f496e0eb1c6d1deebbd277db8aa990c` |
| model-facing producer | `395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf` |
| producer review | `8763602a753345a19312613160d32b3ffe537a7ebfcb4bcf4c83905a25b7ed29` |
| capability compiler | `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390` |
| full logical authority | `191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b` |
| FIT authority file | `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94` |
| FIT record digest | `c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8` |
| compiled contract | `5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2` |
| physical call manifest | `ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e` |
| native metric manifest | `e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf` |

The adapter still binds the remaining closure roles byte-for-byte: result contract
`af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272`, experiment spec
`64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c`, artifact package
`6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc`, battery contract
`b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e`, managed entry
`1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81`, task source
`bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2`, capability preregistration
`da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3`, producer preregistration
`3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882`, compiler review
`3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50`, and all runtime-only source roles
recorded in the final pre-execution review. No closure mismatch was found.

## Complete package and exact price

The generic package validator accepts the final package. It independently binds the receipt to the exact result
bytes, every descriptor to the exact evidence bytes, and the descriptor census to the complete on-disk tree. There
are exactly 24 regular evidence files and no extras or symlinks.

- result SHA-256: `61c0e385dc08ce6f835385e36e17825642af786b706d4af84b8ab9d87958f347`;
- receipt SHA-256: `e7d605a06fcb1b577cfd8bcedc171ea7707f9a4a02e0308f92caddf28688e1e7`;
- canonical evidence-descriptor-list SHA-256:
  `2fb9b6e7a7158c034c4cd4f085f4e8140662b73fc789d63865b52eba7ea2e83f`.

Recompilation from only the captured FIT authority reproduces the frozen contract. Each saved `call.json` is exactly
its corresponding compiled request. The schedule is four base calls followed by four donor calls; every call has
physical shape `21 x 8`. Each of the 16 arrays is finite, C-contiguous `float32[21]`. Rejoining call row IDs,
side-specific target/foil manifests, transform IDs, and array offsets produces exactly 168 unique registered
primitives: 84 base and 84 donor, with 21 observations in each of the eight side-by-transform cells.

The literal price is therefore exactly 8 native forwards, 168 row-side evaluations, zero backwards, zero updates,
and `16 * 21 * 4 = 1,344` raw numeric bytes. NPY headers and call JSON are metadata. The result records only `FIT`,
an empty forbidden-phase list, and `later_phase_generation=false`; every call and every authority row is FIT. No
SELECT, TEST, or OOD authority or artifact is in the reconstructed closure.

## Independent metric reconstruction

I recomputed strict success as `answer_logit > max_foil_logit` and each margin as their float64 difference directly
from the saved float32 arrays, without reading a saved aggregate projection. Results are:

| Side/cell | Strict successes | Accuracy | Mean margin |
|---|---:|---:|---:|
| base, all | 84/84 | 1.000000 | +3.216082755 |
| base:A1 | 21/21 | 1.000000 | +3.235895134 |
| base:A2 | 21/21 | 1.000000 | +3.235895134 |
| base:P | 21/21 | 1.000000 | +3.156645616 |
| base:C | 21/21 | 1.000000 | +3.235895134 |
| donor, all | 74/84 | 0.880952381 | +3.212194085 |
| donor:A1 | 21/21 | 1.000000 | +3.100055536 |
| donor:A2 | 12/21 | 0.571428571 | +0.684748604 |
| donor:P | 20/21 | 0.952380952 | +2.834721270 |
| donor:C | 21/21 | 1.000000 | +6.229250931 |

The registered donor-wide bar is at least 0.90, requiring 76/84, and the cell bar is at least 0.85, requiring 18/21.
Both donor-wide accuracy and donor:A2 fail. Both side-wide mean margins remain positive; all other cells pass. The
single donor:P error (`a92014ebae2de91da0da6e6a5bfd590ffd0c8f90d3d0ae9d3f5cbd90b1c91f4`, target ` red`,
margin -0.037092209) is not needed for the verdict: donor:A2 alone is six successes short of its cell bar.

Reconstructed primitives passed the metric-evidence predicate and reproduced the saved decision exactly:
`native_capability_gate=false`, terminal `hard_abort`, and all seven projection fields null. Publishing the above
audit arithmetic does not retroactively populate those registered result fields or open a continuation.

## Frozen row/token diagnosis of donor:A2

Every FIT sequence is exactly eight jointly tokenized prompt tokens. In A2 the final content positions are
`old_target, new_target, new_target`, and the scored one-token continuation is `new_target`; it is therefore a strict
local previous-token repetition test with an older conflicting target still visible. The nine failed rows are:

| Row ID | Old -> target token | Registered foils (token IDs) | A1 margin | A2 margin |
|---|---|---|---:|---:|
| `6252257ae78db9b5abe8d227140c0aa310c9df091f988ca009192f052afa02ad` | arrow -> dawn (`17577`) | east `7627`, spark `9009`, arrow `15452` | +4.105839 | -0.252262 |
| `aafc4ceb9939f1a29845fcd63748620cdb56906e922ef0c541b7b769abf0a269` | autumn -> road (`2975`) | stone `7815`, river `7850`, autumn `23608` | +0.064324 | -1.249949 |
| `209e929d451616fd02d1f5e7176222573cc20283025c7e8bb975ccb2dde5a3f0` | clock -> white (`2330`) | road `2975`, clock `8801`, autumn `23608` | +2.407759 | -0.005196 |
| `7c78e22bbe62fb7744278fb3020513d03e66591dafb367738fd892edae1c2efb` | copper -> snow (`6729`) | white `2330`, copper `15317`, coral `29537` | +1.464586 | -1.178452 |
| `8bf52aebf02beabdf6404b0948aec8a1e5c355339afaeb02dbce755f542a9a0a` | east -> spark (`9009`) | soft `2705`, gold `3869`, east `7627` | +1.290892 | -2.113928 |
| `4c550708e664eda6a84604d67d1343c3fad072a5bbffe87ceb6ba13bd0a90cb2` | gold -> soft (`2705`) | gold `3869`, island `7022`, lemon `18873` | +1.789662 | -0.953215 |
| `32c781a482544bb17d6d2ed947a22331f4f3434a7d81e1bc33fb4f62be0c4997` | spark -> storm (`6388`) | soft `2705`, moon `8824`, spark `9009` | +2.954074 | -0.425102 |
| `5e51311395d2b50e463f83ee04059e8602ddd73ec45dcc0e7aaf2fa10c30ce9d` | storm -> gold (`3869`) | storm `6388`, island `7022`, moon `8824` | +1.489652 | -0.230463 |
| `c307a37bbae2bdd884cf4356d74ab6d07f0f339dcea1618d73df15fcee945102` | white -> coral (`29537`) | white `2330`, road `2975`, maple `31377` | +3.803407 | -0.105589 |

This pattern is not explained by a changed or degenerate foil set. Within every one of the 21 panels, donor:A1 and
donor:A2 use exactly the same target token and exactly the same three foil token IDs. A1 presents three consecutive
new-target occurrences; A2 changes only the first of those three positions back to the old target. All 21 A1 rows
pass, while nine A2 rows fail and the paired A2-minus-A1 mean-margin change is -2.415306931. The authority balances
every lexical item once in every semantic role, every target is a stable one-token continuation, and each target is
excluded from its nonempty foil set.

Thus the evidence supports the narrow diagnosis the preregistration was designed to expose: two latest repetitions
are not uniformly sufficient in the presence of an older conflicting target, whereas three latest repetitions are.
The shared phrase `Repeat exactly:` is not a formal natural-language command specifying a next-token operation, but
the preregistration explicitly defines this assay as trailing-repeat continuation rather than instruction following;
the prompt and target relation are unchanged and mechanically unambiguous across the panel. Treating the A2 prompts
as too difficult now would be a posthoc task change, not an instrument correction.

The evidence retains only the maximum foil logit, not the identity of the foil attaining it. Therefore this audit
cannot say that the old target, rather than one of the two fillers, won any failed row. That attribution would require
new unregistered evidence or a rerun and is intentionally not made here.

## Package file hashes

Every entry below was rehashed from disk and matched both the result and receipt.

| Evidence path | Bytes | SHA-256 |
|---|---:|---|
| `calls/0000_FIT:base:0:native_base/answer_logit.npy` | 212 | `af1679839ccfbc3e3fcced78a1b339a939243e8cc7d5527e7ef7627704e753bb` |
| `calls/0000_FIT:base:0:native_base/call.json` | 2063 | `40e755dcb9153a7ad5396e007d4563b16098fd5ec3cf240930f2c88f9f79c55e` |
| `calls/0000_FIT:base:0:native_base/max_foil_logit.npy` | 212 | `237481d3e5319164b78b41bef5105f30b036f4581edaa41907478bfd8048c753` |
| `calls/0001_FIT:base:1:native_base/answer_logit.npy` | 212 | `9f275f9dfcb645c85738b47235228d535cce3e8057b20eae07b4e42a911f1424` |
| `calls/0001_FIT:base:1:native_base/call.json` | 2063 | `243f7ca5401b5cf7707e0d4ac3e2dfd2205edf60145fc41c67e636bd88c6865` |
| `calls/0001_FIT:base:1:native_base/max_foil_logit.npy` | 212 | `df9c94e8c6ab625b601f048e66506572b479acfa4bdc9d500b511ded0efe9ac4` |
| `calls/0002_FIT:base:2:native_base/answer_logit.npy` | 212 | `af1679839ccfbc3e3fcced78a1b339a939243e8cc7d5527e7ef7627704e753bb` |
| `calls/0002_FIT:base:2:native_base/call.json` | 2063 | `c57f774950521d1a232f5eff21cea01b1fb227c957a84e5012b03f48654d9609` |
| `calls/0002_FIT:base:2:native_base/max_foil_logit.npy` | 212 | `237481d3e5319164b78b41bef5105f30b036f4581edaa41907478bfd8048c753` |
| `calls/0003_FIT:base:3:native_base/answer_logit.npy` | 212 | `9f275f9dfcb645c85738b47235228d535cce3e8057b20eae07b4e42a911f1424` |
| `calls/0003_FIT:base:3:native_base/call.json` | 2063 | `c2f64fb37971ffe14d61e445015dfab3690108456434521a9f51520430ee3080` |
| `calls/0003_FIT:base:3:native_base/max_foil_logit.npy` | 212 | `49c3b9720729ce591dffb05a61c3b00573e1938efd1f1f9e5a453b43ea7b50f1` |
| `calls/0004_FIT:donor:0:native_donor/answer_logit.npy` | 212 | `2f023f08ab813e50168257ea2a18089b777aeca7e58328f83d2616fbb1c2914d` |
| `calls/0004_FIT:donor:0:native_donor/call.json` | 2067 | `2744b85de4e862866c697e461753383ea46275936189cbbc85a3a26abb56fadb` |
| `calls/0004_FIT:donor:0:native_donor/max_foil_logit.npy` | 212 | `6c3297aa481b702b40b8de8d6741becaa7171c9aba23a5f86b3793555ff342f9` |
| `calls/0005_FIT:donor:1:native_donor/answer_logit.npy` | 212 | `cf269d6c0ae14ce869a995c37810caf1ab394c6b0d0c07646afec3768fdb4de5` |
| `calls/0005_FIT:donor:1:native_donor/call.json` | 2067 | `8e671f7b33bc5e4414c2540a9bd95b60e9e5f5d2f56da69a43ffd79c9a065146` |
| `calls/0005_FIT:donor:1:native_donor/max_foil_logit.npy` | 212 | `82448d2aa053c226413eb921444bbb02d3050cac3c6f5b783b1cfa436a8f791e` |
| `calls/0006_FIT:donor:2:native_donor/answer_logit.npy` | 212 | `fb5096e3ebccaf6e1a9b28064955fd04a8ff13e69f4a6064ca72449c6413eb40` |
| `calls/0006_FIT:donor:2:native_donor/call.json` | 2067 | `6b643dcb5c65bb98a7b4048ba7abdcb831fc3e43c5a486484b8eee76b550a53d` |
| `calls/0006_FIT:donor:2:native_donor/max_foil_logit.npy` | 212 | `eb727b5a4f9aac7de3c7faa35ce3c0d02cab235c00efa328c286236c45ae3526` |
| `calls/0007_FIT:donor:3:native_donor/answer_logit.npy` | 212 | `80f073cad44cc1fc3a10d60886a103289505e6512413c39158dc225746d68be0` |
| `calls/0007_FIT:donor:3:native_donor/call.json` | 2067 | `5e4d511fe4adf504fc01e7fbb9b78851ab1beb226e6443a6c29f52084bd8b2fa` |
| `calls/0007_FIT:donor:3:native_donor/max_foil_logit.npy` | 212 | `397afb06049311e5eaa24beec19d9e0a56f1c26783bffb432017c5e6fb09144e` |

## Review checks

The model-free semantic/compiler suite passes `20/20` with bytecode and pytest cache disabled. A separate read-only
audit script validated the final package, recompiled the contract, reconstructed every primitive, compared the
recomputed decision to the result, and recomputed every statistic above. I did not run producer dryrun against the
now-occupied final namespace because its prospective unused-namespace guard must correctly reject after publication;
that expected stop says nothing against package validity.

## Next strict task recommendation

Close task 21 at this valid negative and make **repaired task 14, subject–verb agreement**, the next strict task—do
not build or run it under this audit. It is a materially different, nonlocal grammatical dependency with archived
capability evidence, so it tests the pipeline without weakening task 21. Its prospective authority should:

1. make A1 flip head-noun number in a prepositional-phrase template;
2. make A2 independently flip head-noun number in a frozen relative-clause template so the answer truly changes;
3. make P change only attractor lexical identity while holding attractor number fixed;
4. use unequivocal coordinated subjects for C and exclude collective/dialect-dependent nouns;
5. jointly verify single-token ` is`/` are` answers and freeze exact per-phase authorities with the same capability-first,
   all-null-on-fail boundaries.

Proceed only if those repairs make head noun, attractor, surface form, lexical roles, and answer token independent and
balanced. Task 21's observed failure supplies no license to reuse its rows, change its thresholds, or localize it.
