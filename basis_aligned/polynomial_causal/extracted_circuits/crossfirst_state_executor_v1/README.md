# Explicit-state conditional crossfirst executor — implementation awaiting native replay

This assembles the already tested all-source interaction from frozen weights and native normalized state inputs. It computes Q7, full joint head8.2 routing, first-layer token values, and selected even joint head9.8 routing internally. It accepts neither donor prompts nor city annotations nor cached Q/attention patterns.

Required contextual inputs are the native normalized MLP7, attention8 and attention9 states, plus squared RMS8 and RMS9. The native prefix that generates those inputs and the downstream suffix remain external. This is a more explicit conditional implementation, not an autonomous circuit or smaller replacement model.

The implementation reuses the frozen token generator and selected head9 routing helper; its head8 routing retains both full QK factors. Native replay of the assembled executor is **pending**. Do not replace the previously validated executor or claim adoption until that replay passes.

External checkpoint matrices are the57,950,208-scalar embedding and10,616,832-scalar MLP7 left/right weights, in addition to the folded reader and routing programs loaded by `load_weights`. Dependencies are exposed in code rather than omitted from the price. This first assembly prioritizes correct computation; double-precision conversions and lack of caching are not a measured speedup.

Next validation: capture the three normalized states and two norm ports on the existing96regional and64FineWeb prefixes, compare generated fields against the independent hook-based native computation, then replay the resulting removal. Preserve every prior failed generalization/control criterion.
