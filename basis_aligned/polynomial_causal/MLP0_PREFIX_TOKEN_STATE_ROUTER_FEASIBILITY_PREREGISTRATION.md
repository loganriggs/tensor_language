# MLP0 four-state prefix-token router feasibility — preregistration

Date: 2026-09-01 16:36 UTC

## Decision and scope

Rung407 rejected a document-level router because even its future-label oracle lost to a cheaper rank768 program.
However, its separately executed position oracle showed large I-active/Fisher complementarity. This rung asks whether
a genuinely small state computed only from the available prefix can predict enough of that complementarity to
justify one physical routed-forward experiment.

This is an off-policy feasibility screen over immutable saved losses. It is not a coherent routed forward,
compression, or adoption. A pass licenses exactly one physical two-expert router gate; a miss does not permit feature,
tree-size, pair, or threshold tuning.

## Frozen authority and split

- Losses: rung407's `mlp0_p448_router_oracle_losses.pt`, exact tensor SHA256
  `e6d92614ad4fbe5b6e63aa2939e7df6ecb197281c114aae9696baf3bc68ab082`, shape `8x384x192`.
- Experts: the already fixed `I_active_p448` and `Fisher_p448`; no pair selection in this rung.
- Tokens: rung407's exact 384 source-document rows and positions `[64:256)`.
- Train documents: source ordinals `[0:192)`.
- Evaluation documents: source ordinals `[192:384)`, reported as two contiguous96-document waves.
- Native and covariance-p768 losses remain the price/damage references. FINAL remains unopened.

## Prefix-observable features

For scored position `p` in a document, the current token and prefix through `p` are available; the target token and
future prefix are forbidden. Freeze these features before reading expert-win labels:

1. absolute position divided by255;
2. current-token GPT-2 byte morphology, one-hot over four states:
   - word-start alpha: leading ASCII space followed by an ASCII letter;
   - continuation alpha: first byte is an ASCII letter with no leading space;
   - digit: any ASCII digit byte, taking precedence over the alpha states;
   - other;
3. the previous token's same four-state morphology;
4. whether the current token occurred earlier in the prefix;
5. `log(1+distance_to_previous_occurrence)/log(258)`, using distance257 when never seen;
6. `log(1+training_frequency(current_token))/log(1+max_training_frequency)`, where frequencies use only input
   tokens from train documents0:192.

The model receives 12 numerical columns: position, repeat flag, normalized log distance, normalized log frequency,
and two four-way one-hot morphology vectors. No loss, target token, future token, native activation, document ID, or
evaluation statistic enters a feature.

## Frozen state models

The target on training positions is

`advantage = loss_Fisher - loss_I`.

Positive predicted advantage selects I-active; negative selects Fisher-active.

Primary model: scikit-learn `DecisionTreeRegressor(random_state=408, max_leaf_nodes=4, min_samples_leaf=2048)`, fit
once on all train positions. Four leaves are the maximum four router states, not four experts.

Matched controls, each assigning its expert from training mean advantage only:

- constant I-active;
- four fixed position quartiles;
- four current-morphology states;
- four repeat-distance states: never, distance1--8, distance9--32, distance33--256.

All models are evaluated by selecting the corresponding immutable expert loss at each evaluation position. This is
still off-policy because the saved losses came from complete single-expert executions.

## Measurements

- evaluation damage above saved native for I, Fisher, p768, the I/Fisher future-label oracle, every control, and the
  four-leaf tree;
- fraction of the available I-to-oracle gain recovered by each state model;
- two heldout-wave damages, expert-use fractions, leaf counts/supports, tree structure, selection accuracy, and
  loss-weighted regret;
- literal optimistic program price14,599,296 values plus explicit tree/state price, compared with p76813,272,192.

## Frozen predictions

### A — authority, split, and state instrument are exact

- loss tensor/hash/names, row receipt/hash/order, scored positions, and train/evaluation disjointness are exact;
- all features/losses are finite; training frequency uses no evaluation rows; target/future-token perturbation checks
  leave features unchanged;
- the tree has at most4 leaves, every leaf has at least2048 training positions, and evaluation changes both selected
  expert and routed loss relative to constant I; FINAL is unopened.

### B — heldout position oracle retains price-relevant headroom

On evaluation documents, I/Fisher position-oracle damage is at least `0.0002 nat` lower than p768 damage.

### C — four prefix states recover enough headroom

- tree recovery fraction is at least25%;
- tree damage is at least `.001 nat` lower than constant I;
- tree damage is at least `.0002 nat` lower than p768.

### D — the gain transports and uses both experts

- tree damage is lower than p768 in both96-document evaluation waves;
- each expert is selected on at least10% of evaluation positions;
- every tree leaf has at least2048 training positions.

## Strong null

The strong null fires if A or B fails; tree improvement over constant I is below `.0002 nat`; tree damage is not
lower than p768; or either expert is selected on fewer than2% of evaluation positions.

## Decision

- A+B+C+D and no null: freeze this exact tree and run one coherent physical token-routed forward, including exact
  two-expert/tree/state price, p768, constant-expert, shuffled-leaf, OOD, and signed-intervention controls.
- Oracle headroom with a tree miss: the complementarity is not captured by this cheap interpretable state. Close
  prefix-token routing among these experts without trying larger trees or learned neural routers.
- Strong null: close the route and advance direct nonlinear CE fitting or an output-side representation change.
