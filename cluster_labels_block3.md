# Cluster labels: depth-gated tokens, gated_block3 / block3-seed0

Source: `/workspace/tensor_language/runs_lm/cfp_report_gated_block3_block3-seed0.json`
Model: 6-layer bilinear block3 (L0=attn×4, L1=MLP, L2=attn×4, L3=MLP, L4=attn×4, L5=MLP), TinyStories, byte-BPE 1024.
Gate: tokens with block3 median CE < 0.25 nats and block2 median CE > 1.5 nats. 2500 sampled, k-means (k=8) over knockout fingerprints, load-bearing threshold ΔCE > 0.5.

## Global component signature (read this first)

| cluster | n | ind_frac | L1MLP | L3MLP | L5MLP | strong heads (ΔCE) |
|---|---|---|---|---|---|---|
| 0 | 671 | 0.57 | 9.8 | 9.9 | 3.4 | none (L4H1 0.69, L4H3 0.65) |
| 1 | 386 | 0.38 | 4.8 | **11.7** | 3.1 | none (L4H0 0.89) |
| 2 | 485 | 0.59 | **13.8** | 5.4 | 3.2 | none (L4H1 0.76) |
| 3 | 307 | 0.55 | 8.8 | 5.4 | **6.0** | none (L0H0 0.89) |
| 4 | 167 | 0.61 | 7.8 | 8.0 | 3.8 | **L2H0 5.4** |
| 5 | 106 | 0.81 | 9.7 | 7.1 | 3.6 | **L0H2 3.9 + L2H3 3.5** |
| 6 | 255 | 0.49 | 4.2 | 7.6 | **5.7** | L0H0 1.5 |
| 7 | 123 | **0.14** | 7.6 | 7.3 | 4.2 | **L4H0 5.5** |

Every cluster shares the L1MLP + L3MLP + L5MLP backbone (the known bilinear-MLP "statistics circuit" stack: L1 ≈ detokenize fragment→word-identity features, L3 ≈ order-2/3 context statistics over those features, L5 ≈ re-tokenize word features→next-BPE-piece). What separates clusters is (a) the *ratio* of L1:L3:L5 and (b) whether any attention head is genuinely load-bearing. Only three clusters have a head above 1.5 nats — C4 (L2H0), C5 (L0H2→L2H3), C7 (L4H0) — and these are the three mechanistically distinct circuits. Clusters 0–3 and 6 are almost certainly one MLP-stack family that k-means sliced along loading ratios.

Head-role inventory implied by the table: L0H0 is a ubiquitous weak previous-token/local head feeding the backbone everywhere; L0H2 is a previous-token head paired specifically with L2H3 (classic prev-token→induction composition); L2H0 and L2H3 are two layer-2 induction heads with different niches (names vs. article+fragment nouns); L4H0 is a *deep* match head that runs after two MLPs and behaves unlike classic induction (see C7); L4H1/L4H3 are weak cleanup contributors.

---

## Cluster 0 — n=671, ind_frac 0.57

**LABEL: Balanced-MLP word-internal completion of common words (copy-optional lexicon completion).**
Examples are mid-word continuations of frequent TinyStories words: `wra→p`, `g→un` (gun earlier), `co→c(oa)` (cocoa earlier), `tablec→l(oth)`, `monkey→s` (monkeys earlier), `Dr. S→m(ith)` (Smith earlier), `st→one` (stone earlier), `Da→v(id)`, `free→z`, `n→et`, `pus→h`. About half the targets do appear earlier in context (ind_frac 0.57), but the fragments here are long/distinctive enough that the continuation is largely determined by spelling statistics alone — the earlier occurrence is corroborating, not necessary. L1MLP and L3MLP are equally and enormously load-bearing (~10 nats each), heads are negligible.

**HYPOTHESIS:** This is the plain stacked-MLP lexicon circuit: L1MLP maps the current word-so-far (fragment BPE pieces, delivered locally by L0 heads) into word-identity features; L3MLP sharpens/selects using 2–3 tokens of left context; L5MLP emits the next piece. Block2 fails not because it lacks the mechanism but because one MLP pass is insufficient — the completion needs the L1→L3 *composition* (two multiplicative lexicon lookups in series). The weak L4H1/L4H3 contributions are the copy pathway kicking in on the ~57% of examples where the word occurred earlier. Shares its backbone with C1/C2/C3 — same circuit, different loading (see summary).

**INTERVENTION:** Context-truncation test, no ablation needed: re-run the model on these 671 tokens with context truncated to the last 8 tokens (word fragment + a few words). Prediction: median P(target) stays high (CE < 0.5) for most of the cluster, proving the circuit is local-statistics, not retrieval. On the subset where truncation hurts, check L4H1/L4H3 attention offset — they should attend near the earlier occurrence of the word; zeroing L4H1+L4H3 *only at the final position* should hurt only that subset (ΔCE > 0.5) and leave the truncation-robust majority untouched.

---

## Cluster 1 — n=386, ind_frac 0.38

**LABEL: Context-selected completions and collocations — L3MLP-dominant (order-3 statistics beyond spelling).**
Two intertwined example types: (a) word completions where the fragment alone underdetermines the target and 2–3 words of left context select it: `ac→c(ident)` after "a minor", `sk→ill`, `vo→ic(e)` after "have their", `s→ch(ool)` after "start", `stethos→c`; (b) multi-token collocations where the target is a *new word* chosen by an idiom: `in a calm→ v(oice)`, `plenty→ of`, `never seen the sea→ before`, `put→ out` (fire), `sa→n(g)`. Lowest MLP-family induction fraction (0.38): the answer typically is NOT a copy of anything earlier; it's corpus statistics. L3MLP towers at 11.7 while L1MLP drops to 4.8.

**HYPOTHESIS:** Same backbone as C0, but the decision is made at L3: L1 provides candidate word-identities, and L3MLP performs the order-3 n-gram/collocation selection over them ("calm __" → voice; "plenty __" → of). This is the purest form of the known "statistics circuit" where the bilinear MLP implements a lexicon of multi-word collocations. Depth-gated because the collocation lookup needs word-level features that only exist after L1 — block2's L3 sits too shallow after a weaker L1 to hold both the detokenizer and the collocation table.

**INTERVENTION:** Corpus-statistics check: train a trigram/4-gram model at the BPE level on TinyStories and measure how often the cluster target is the n-gram argmax (expect >70%, vs. much lower for C5/C7). Then the causal side: zero L3MLP *only at the final position* (leave it intact everywhere else). Prediction: CE on these tokens jumps by nearly the full 11.7 nats — i.e., L3's contribution here is a final-position readout computation, not upstream context processing. If instead CE only rises when L3 is zeroed at *earlier* positions too, the collocation is being cached in context word representations, which would be a different (more interesting) story.

---

## Cluster 2 — n=485, ind_frac 0.59

**LABEL: Discourse-noun re-completion after short cue — L1MLP-dominant ("the c…" → earlier-mentioned word, easy cases).**
The dominant template: an article/possessive + 1-piece fragment whose completion is a noun (or name) already introduced in the story: `Bru→ce` (Bruce earlier), `a p→as(sport)`, `the c→ake`, `the c→ir(cus)`, `O→ll(ie)`, `the m→a(ze)`, `st→u(dents)`, `bloss→om`, `the p→u(zzle)`, `the c→age`, plus regular morphology fillers (`chas→ed`, `choos→es`, `perfect→ly`, `child→re`). Highest L1MLP loading in the whole report (13.8); heads negligible.

**HYPOTHESIS:** Same circuit family as C5 (see below) but the *easy* regime: the story is short and topically narrow enough that the fragment + a couple of context words statistically determine the completion, so the MLP lexicon stack answers without needing an explicit attention retrieval — the "induction-looking" ind_frac 0.59 is largely epiphenomenal (in TinyStories the completed word almost always did occur earlier). L1MLP dominance suggests the current-token fragment identity itself carries most of the answer. k-means separated this from C5 precisely because in C5 the fragment (`the p`) is ambiguous and the induction heads become load-bearing, while here they are not.

**INTERVENTION:** Referent-swap test on ~50 examples: edit the context, replacing all earlier mentions of the target word with a different same-initial word (e.g., every "cake" → "cave", leaving the final `the c` cue intact). If this cluster truly runs on MLP statistics rather than retrieval, P(original target) should stay relatively high / the prediction should NOT cleanly flip to the swapped word — in sharp contrast to C5, where the same manipulation should flip the prediction. A clean flip here would falsify the "epiphenomenal copy" reading and merge C2 into C5's circuit.

---

## Cluster 3 — n=307, ind_frac 0.55

**LABEL: Article + fragment → earlier-mentioned noun, with heavy L5 output-side selection ("the b/st/sh…" cases).**
Essentially the same template as C2 — `the b→ar(rel)`, `the st→ream`, `the b→and`, `your c→am(era)`, `the p→in`, `the sh→ark`, `a bad→ge`, `che→es(e)`, `the me→ad(ow)` — but the knockout profile shifts weight to L5MLP (6.0, its second-highest anywhere) with L1 and L3 both moderate. Fragments here (`b`, `st`, `sh`, `p`) tend to be shorter/more ambiguous than C2's, and several targets are the *second* piece of a longer word (`ac→ce`, `acc→om`), i.e., the model must commit to a specific multi-piece spelling.

**HYPOTHESIS:** Same discourse-noun re-completion circuit as C2/C5, caught in the regime where the *output selection* is hard: several candidate words are compatible with fragment+context, and L5MLP performs the final winner-take-all mapping from the resolved word feature to the exact next BPE piece. I.e., C2 vs C3 is not two circuits but two bottlenecks of one circuit (input detokenization vs. output re-tokenization). The weak-but-present L0H0/L4H0 are the residue of the retrieval pathway.

**INTERVENTION:** Position-specific L5 swap: zero L5MLP at the final position only — CE should jump ~6 nats here but only ~3 in C2 (matching the fingerprints and confirming final-position locality). Sharper: logit-lens at pre-L5 residual — decode the residual stream after L4 on C3 examples. Prediction: the correct *word* is already resolved (earlier-mention word features dominate; e.g., nearest-neighbor word embedding = "barrel") but the correct *piece* (`ar`) is not yet top-1 in the token basis, becoming top-1 only after L5. That directly shows L5 = word→piece re-tokenizer rather than decision-maker.

---

## Cluster 4 — n=167, ind_frac 0.61

**LABEL: Name/proper-noun induction across sentence boundary — L2H0 induction head.**
Examples overwhelmingly complete a *capitalized* token that starts a sentence or vocative, where the name/noun appeared earlier: `\n\nN→ick` (Nick earlier), `T→on(y)` (Tony earlier, twice), `Mrs→.`, `Mr. Firem→an`, `the t→ur(tle)`, `toy o→st(rich)`, `the b→ill(board)`, `the f→il(m)`, `the→ we(ll)`. Single capital letters ("N", "T") are maximally ambiguous fragments — pure statistics cannot pick Nick vs. Nancy — so retrieval is mandatory. L2H0 is strongly load-bearing (5.4 nats), the only cluster where it appears at all.

**HYPOTHESIS:** Classic two-layer induction specialized for names: some L0 head (likely L0H0, present at 0.69–0.89 across the MLP clusters as the generic prev-token head) writes previous-token info; L2H0 does the match-and-copy from the position after the earlier occurrence of the fragment; L1/L3 MLPs supply the fragment-identity features L2H0 matches on, and L5 re-tokenizes the retrieved word to the next piece. Depth-gated relative to block2 presumably because block2's L2 attention must do this *without* a preceding strong L1 detokenizer output at match quality, and lacks the L5 re-tokenizer entirely (block2 ends at L3). L2H0 (names) vs. L2H3 (C5's "the p" nouns) look like two induction heads that carved up the retrieval domain.

**INTERVENTION:** Attention-offset check + targeted ablation: on the 167 examples, record L2H0's attention distribution from the final position. Prediction: >50% of mass lands at offset +1 after earlier occurrences of the same fragment token (the "ick" after the earlier "N", the "on" after the earlier "T"). Then zero L2H0 *only at the final position*: CE should recover ~5 nats of the knockout effect, confirming the effect is final-position retrieval, not context preprocessing. Cross-check specialization: apply the same final-position L2H0 ablation to C5's examples — it should do almost nothing there (and vice versa for L2H3 on C4).

---

## Cluster 5 — n=106, ind_frac 0.81

**LABEL: Fragment-cued discourse-referent retrieval, hard/ambiguous regime: "the p" → the story's p-word (pie/pony/pig/purse/paint/pillow/pilot/palace/package/person…) — L0H2 → L2H3 prev-token→induction pair.**
The cleanest cluster in the report: nearly every example ends `the p` / `my p` / `little p` and the target is the continuation of whichever p-noun that particular story introduced: `p→ie`, `p→on(y)`, `p→ig`, `p→ur(se)`, `p→ain(t)`, `p→ack(age)`, `p→ill(ow)`, `p→il(ot)`, `p→al(ace)`, `p→ers(on)`. The letter "p" happens to be BPE-atomic after "the " with a huge candidate set, so the fragment carries ~zero information — the answer *must* be retrieved from context. Correspondingly: highest ind_frac (0.81) and a textbook composition signature, L0H2 (3.9) + L2H3 (3.5), unique to this cluster.

**HYPOTHESIS:** L0H2 is a previous-token head; L2H3 is its induction partner: at the final "p", L2H3 matches against L0H2-shifted keys to find the earlier "…the p|X…" position and copies X's identity into the residual; L1/L3 MLPs both feed the match (fragment features) and integrate the retrieved word; L5 re-tokenizes. This is the same *task* as C2/C3, but k-means correctly isolated the subpopulation where the MLP-statistics shortcut is impossible and the attention circuit is causally necessary. Depth-gating: block2 owns an L0→L2 pair too, so the gate is probably the missing L5 re-tokenizer plus L3-assisted matching — worth checking against block2's actual failure mode on these tokens.

**INTERVENTION:** Referent-swap (the decisive one): in each context replace the earlier p-word with a different p-word (pony→pig everywhere upstream), keep the final `the p` cue. Prediction: block3's top-1 flips to the swapped word's continuation on the large majority of examples — proving retrieval, not statistics. Mechanistic confirmation: L2H3 attention from the final position should concentrate at offset +1 after earlier `p`-fragment positions, and zeroing L2H3 (or L0H2) at the final position only should send the prediction to the generic p-continuation prior (roughly corpus P(·|"the p")) with CE jump ≈ 3.5 nats.

---

## Cluster 6 — n=255, ind_frac 0.49

**LABEL: Morphological suffix and grammatical-slot selection — inflections (-s/-es/-ed/-ing/-ily/-ier) and grammar-bound continuations, L3+L5-heavy, L1-light.**
Targets are dominantly *suffixes* chosen by syntax rather than word identity: subject agreement `run→s` ("She … runs"), `get→s` ("no one gets"), tense `calm→ed`, adverbialization `happ→ily`, `happ→i(er)`, `bruis→es`, plus grammatical multi-token frames `not as heavy→ as`, `side to s→ide`, `paid her any att→ent(ion)`, `kneel down b→es(ide)`, `grass ben→e(ath)`, sentence-final `too→.`. Uniquely low L1 (4.2) with high L3 (7.6) and L5 (5.7); L0H0 at its strongest (1.5).

**HYPOTHESIS:** A late-layer *grammar* circuit distinct from the lexicon path: the stem's identity matters less than its syntactic environment (subject number, tense of the surrounding narrative, comparative frames). L0H0 pulls in the local syntactic neighbors (subject, auxiliary), L3MLP computes the agreement/tense decision from them, and L5MLP spells the chosen suffix. Low L1 loading = the circuit doesn't need deep fragment detokenization, consistent with suffixes being about categories, not words. This is plausibly semi-new relative to "statistics circuits": it's conditional inflection, not n-gram lookup — though order-3 statistics can mimic much of it in TinyStories.

**INTERVENTION:** Agreement-flip minimal pairs: take the subject-agreement examples (`run→s`, `get→s`) and edit only the subject's number ("She stops … and run" → "They stop … and run"; "no one" → "they"). Prediction if the hypothesis holds: P("s") collapses and P of the bare/plural continuation rises — i.e., the circuit tracks syntax, not a memorized trigram (a trigram model sees "and run" unchanged and would NOT flip). Pair with final-position-only L3MLP vs. L5MLP zeroing: L3 knockout should destroy the *choice* (probability shifts to the wrong suffix), L5 knockout should destroy the *spelling* (mass spreads over many pieces).

---

## Cluster 7 — n=123, ind_frac **0.14**

**LABEL: Fuzzy/semantic retrieval with re-inflection — deep induction on lemma/meaning, not surface bigrams: L4H0.**
The standout facts: lowest induction-pattern fraction by far (0.14 — the strict "earlier bigram (prev-token, target)" template almost never holds) yet the examples clearly depend on distant context, and L4H0 is massively load-bearing (5.5). Example structure: **(a) inflection-changing copy** — earlier "a cap" → `many cap→s` (retrieve lemma, output *plural*, so the surface bigram never occurred); "what are you discussing?" → `We are discuss→ing`; "stop and yield" (infinitive) → later re-use; `included meadow→s`, `soup with crystal→s`; **(b) semantically-associated new words** — "asked her to marry him … On the day of the `w→ed(ding)`" (wedding never appeared; *marry* did); "he saved them … thank Sam for `sa→ving`"; "old lady … The `old→ l(ady)`"; `He wants his a→pp(le)` (possession established earlier as "Tom's apple"); **(c) construction completion** — `dared her friends→ to`, `shared her grapes→ with`, `cheerful→ v(oice)`. Common thread: the answer is recoverable from context only at the level of *lemmas/semantic features*, and often must be re-inflected before output.

**HYPOTHESIS:** L4H0 is a *second-order* induction head: sitting after L1MLP and L3MLP, its queries/keys are built from processed word/lemma features rather than raw token identities, so it can match "cap(sg)" when cueing "cap(pl)", or "marry" when cueing "wedding"-frame — fuzzy induction over the MLP-computed feature space. Pipeline: L0 heads + L1/L3 MLPs detokenize and semantically annotate the context; L4H0 retrieves the matching lemma/associate from long range; L5MLP re-inflects/re-tokenizes it into the correct surface piece (its 4.2 loading here is doing real work: plural -s, -ing, -ed). This is a genuine 3-stage composition (MLP-features → deep attention match → MLP re-spell) that structurally cannot fit in block2: block2 has no attention *after* its second MLP and no MLP after its second attention, so lemma-level match + re-inflection is impossible at any weights — the strongest depth-gating story in the report. Note L4H0 also shows up weakly in C1/C3 (0.68–0.89): the same head lending soft retrieval support to the statistics clusters.

**INTERVENTION:** Three cheap tests, in order of leverage. (1) *Attention check for fuzzy match:* on the inflection-changing examples, measure L4H0 attention from the final position; prediction: mass concentrates on/after the earlier occurrence of the *lemma in a different surface form* ("a cap" tokens when predicting `cap→s`), which a surface-matching induction head would not select. (2) *Final-position ablation:* zero L4H0 only at the final position → CE jump ≈ 5 nats and top-1 falls back to a generic continuation; zeroing it everywhere *except* the final position should do little. (3) *Lemma-swap:* replace "a cap" with "a hat" upstream (keeping `many cap` impossible — instead cue `many h`) — prediction flips to `at`; and for the semantic subtype, replace "marry" with "race" and check P(`ed`|`the day of the w`) collapses. If (1)+(3) hold, this is confirmed as retrieval over abstract features — not induction (surface match fails), not n-gram statistics (target depends on distant unique context).

---

## Cross-cluster summary

**Same circuit, split by k-means:**
- **C0, C1, C2, C3 are one family** — the stacked bilinear-MLP lexicon/statistics circuit (L1 detokenize → L3 order-2/3 select → L5 re-tokenize) applied to word-internal completion. k-means split them along which stage bottlenecks: C2 = input/fragment-heavy (L1-dominant), C1 = context/collocation-heavy (L3-dominant), C3 = output/spelling-heavy (L5-elevated), C0 = balanced. Their ind_frac ≈ 0.4–0.6 is mostly epiphenomenal (TinyStories words recur), which the C2 referent-swap test would confirm. This matches the known finding that word-internal completions dominate depth gates.
- **C4 and C5 are sibling retrieval circuits**, the *same task* as C2/C3 in the regime where the fragment is too ambiguous for statistics: classic fragment-level induction, implemented by two different layer-2 heads that partitioned the domain — L2H0 for capitalized names/sentence-initial referents (C4), L0H2→L2H3 for article+letter noun cues (C5). Known building block (prev-token→induction), but nicely specialized.
- **C6 is plausibly distinct** — suffix/agreement selection driven by syntax with almost no L1 involvement — but could still reduce to order-3 statistics; the agreement-flip minimal pairs decide it.

**Genuinely new candidates:** C7 clearly; C6 possibly.

**Priority: Cluster 7.** It is the only cluster that is neither induction (ind_frac 0.14 — the strict bigram pattern fails by construction, since the retrieved word reappears *re-inflected* or as a semantic associate) nor plain n-gram statistics (targets hinge on distant, story-unique content like "marry"→`wedding`, "a cap"→`caps`). Its fingerprint is unique — L4H0 at 5.5 nats, the only strongly load-bearing layer-4 head anywhere — and its proposed mechanism (MLP-built lemma features → deep attention match → MLP re-inflection) is exactly the kind of 3-stage composition that a 4-layer block2 cannot express under any weights, making it the most principled explanation of depth-gating in the whole gate set. It is also cheap to attack: 123 examples, one head, and three decisive tests (fuzzy-match attention pattern, final-position-only ablation, lemma-swap flip).

---
## Post-hoc causal verdicts (main agent, 2026-07-09)
- C7: single-head-at-q CONFIRMED (ΔP 0.78 vs 0.10 control) but the retrieval semantics
  REFUTED — signed pattern×OV attribution puts the source at offset 1-2 (lemma overlap
  11% ≈ control). Relabeled: "deep local read-off" (L4H0 reads the previous position's
  2-block-deep residual). The depth-gating story stands, via feature depth rather than
  match distance.
