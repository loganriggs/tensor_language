# Subject-number sparse graph token extraction V1

## Deliverable

Create a reusable frozen component whose public data inputs are token IDs,
subject positions, and correct/foil answer IDs. With checkpoint weights and the
frozen embedding-number decoder, it internally constructs the five graph ports
and returns the nine-edge predicted agreement-margin damage. No activation from
an external model run, prompt lookup, term selection, or fitted coefficient is
accepted.

The component evaluates only the ten suffix corners required by the frozen five
main and four pair edges, rather than the full 32-corner census. A separate
all-five corner supplies the causal target for validation. Its exact frozen
masks are `[8,1,4,2,24,12,16,9,20]`.

## Validation

Run the identical component on:

- all 128 rows of the corrected original crossed authority;
- all 32 recipient prompts of the disjoint-vocabulary, new-template,
  length-nine fresh authority.

Report the original four held-out panels and the fresh four
direction-by-template cells. Independently replay the native base logits to
audit the extracted suffix path.

## Registered gates

- Instrument: finite; component module and all parents hash-bound; zero external
  activation inputs; exactly five internally derived ports, nine edges, and ten
  predictive corners; native replay at most `1e-5`; aggregation error at most
  `1e-10`; numerical-gauge correction relative L2 at most `1e-5`; exact prices
  and checkpoint; native accuracy at least `0.75` in every reported cell.
- Original authority: relative L2 at most `0.10`, cosine at least `0.99`, and
  positive aligned recovery in every panel.
- Fresh authority: relative L2 at most `0.15`, cosine at least `0.98`, and
  positive aligned recovery in every cell.

Passing establishes token-input extraction of the sparse graph in the sense of
the briefing. It does not claim computational savings over two native forwards:
the component still uses the checkpoint's exact layers to derive its internal
ports.
