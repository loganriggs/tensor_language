# Controlled-city sources of inherited even values

Use all 48 article-corrected rows. Generate source masks solely from paired token
differences. Keep the complete native context in Q/K and all upstream states.
Partition Rf into even(lambda Vfirst at changed city positions) and its complement.
This localizes value sources, not all routes through which the cue affects routing.

192 native forwards: native, remove Rf_city, remove Rf_other, remove Rf_all.
180 second cap, two shared graph calls per source, no extra model weights.

A: native anchor relative error <=1e-5 versus the corrected full-cube artifact;
source recomposition <=1e-10. B: city-only paired-cue effects approximate Rf_all
within20% relative L2 in each family AND Rf_all norm >=1% full-head cue-effect
norm in each family. C: separate city/other effects sum to Rf_all within5% for
paired cue and individual control readouts in each family. Denominator floor1e-8.
Null: inherited cue effects are distributed across other value positions through
context-dependent routing, or city-only effects fail the materiality gate.

No row filtering, refitting, role-label adoption, general preservation, static
compression or corpus OOD claim. Original weights and supplied native prefix,
first values and suffix remain explicit dependencies. Preserve prior failures.
