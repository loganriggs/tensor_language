# Correction to the upstream port closure

The first preflight added the full MLP7 output to both the RMS8 state and the attention8 numerator. That double-counted the MLP7 contribution because the folded reader already supplies its q1/k1/q2/k2/value numerator channels. The corrected operator adds full MLP7 only to the RMS8 denominator and leaves the reader numerator unchanged.

The corrected result has maximum relative write error `2.0e-6` on 40 prior Pile fixtures and passes the existing FineWeb fixture panel with the same gate. This is an algebraic fold correction, not yet a new fresh behavioral confirmation: the next run must select a new outcome-blind panel and exercise the full coupled suffix with all four gates.
