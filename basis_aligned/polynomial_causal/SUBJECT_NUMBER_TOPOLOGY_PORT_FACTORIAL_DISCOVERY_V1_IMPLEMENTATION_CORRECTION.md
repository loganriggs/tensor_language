# Subject-number topology port-factorial V1 implementation correction

The first execution stopped before producing a result because the validated
`_hybrid_input` helper enumerates the 63 nonempty subsets only. The registered
factorial also includes the empty/native corner, and the runner incorrectly passed
the empty string to that helper instead of directly using the captured recipient
raw state. The failure occurred before selection, metrics, or an output artifact.

The correction special-cases only the empty corner as recipient `raw_state` with
zero hybrid-closure error. All 63 nonempty corners still use the validated helper.
Rows, subset order and hash, folds, selection rule, prices, bars, and all other
code remain unchanged.

The next execution completed the factorial but the standard-JSON writer rejected
the numeric `inf` sentinel used when a selected cardinality had no matched
alternative (cardinality zero or six). The preregistration already says this case
fails the null gate. The runner now serializes the unavailable median and advantage
as JSON `null` and keeps the gate false. This changes no model value, selection,
or scientific rule.
