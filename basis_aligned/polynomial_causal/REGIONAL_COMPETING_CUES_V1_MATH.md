# Competing city cues: what does the extracted branch follow?

The frozen branch remains numerically faithful on 48 new, longer contexts, but
its first competing-cue screen fails the stronger behavioral criteria. These
contexts reuse six spelling concepts and known cities; they test new combinations
and clause orders, not unseen vocabulary or broad corpus OOD.

Each prompt gives an editor a city and a tourist a second city, then continues
the editor's report. Two clause orders cross editor Manchester/Seattle with
tourist Liverpool/Denver. Let $m(t,d)$ be the UK-minus-US spelling margin, where
$t,d$ select the target/editor and distractor/tourist cities. The paired target
and distractor effects hold the other city fixed:

$$
T=\frac12\sum_d[m(\mathrm{UK},d)-m(\mathrm{US},d)],\qquad
D=\frac12\sum_t[m(t,\mathrm{UK})-m(t,\mathrm{US})].
$$

For branch removal use $e=m-m_{\mathrm{removed}}$ in the same expressions.
Coverage is $T(e)/T(m)$; role selectivity separately compares mean absolute
paired target and distractor effects. This uses actual capped native logits,
not an assumed linear readout.

[Native result](REGIONAL_COMPETING_CUES_V1_RESULT.json): numerical A passes,
B (coverage/selective controls) and C (twofold role selectivity) fail. Native
target differences are positive on all24 comparisons, with order means2.17
and1.78nats. Removing the branch reduces them by11.15% and9.59%; the latter
misses the frozen10% criterion. Unrelated contrast changes remain small.
The tourist-city effects are also large: native2.11/2.34nats and component
0.225/0.277nats versus target component0.242/0.171nats. Neither native nor
component has the required twofold target advantage.

[Executed full-component check](REGIONAL_COMPETING_CUES_V1_ACCOUNTING.json)
finds11.15/9.60%coverage when all nine shared-component consumers are retained.
Dropping the other consumers therefore does not explain the coverage failure.
Mixed-package versus original head2 write error is6.40e-8; child error4.19e-7.
Precision or package execution is likewise not a plausible explanation here.

A methodological limit remains: city identity is confounded with semantic role.
The [registered crossover](REGIONAL_CITY_ROLE_CROSSOVER_V1_PREREGISTRATION.md)
swaps the city-pair assignments in48otherwise corresponding contexts. It is a
new prospective test under unchanged bars; the first B/C misses are preserved.
Combine both assignments before claiming role-insensitive regional priming.
No fitting or selected-row removal is permitted.


## Completed crossover and child accounting

The [city-role crossover](REGIONAL_CITY_ROLE_CROSSOVER_V1_RESULT.json) passes
A/B and fails C. Coverage is12.62/11.31% in its two orders; numerical child error
is at most5.30e-7. Together the two panels comprise96contexts and48target
contrasts, all of which show a positive joint-component removal effect.
This does not repair the original9.59%coverage miss.

[Balanced role accounting](REGIONAL_CITY_ROLE_CHILD_ACCOUNTING_V1_RESULT.json)
averages each city pair over both roles. Native target/distractor response
ratios are1.350 when the editor clause comes first and0.949 when it comes
second. The joint component ratios are1.635 and0.939. The absence of a robust
twofold editor advantage therefore persists after city-pair counterbalancing.
This supports a regional-cue contribution with substantial distractor sensitivity
on this panel; it does not establish global role blindness or a single lexical
lookup as the complete mechanism. Clause order still changes the computation.

Both existing child branches carry positive editor-city effects, with child0
larger. Child0 role ratios are1.749/0.995 and child1 ratios1.234/0.787. Neither
child alone supplies a robust role-specific split under this screen. The child
programs remain independently executable; semantic exclusivity does not follow
from their algebraic decomposition.

Using the mixed joint arm and its own child arms, the final-margin nonadditive
remainder is0.210–0.215% of the joint removal-effect norm. Unlike the inherited
precision-run interaction statistic, this compares branches within the same
package. It supports approximate behavioral addition on these96contexts,
not a general composition guarantee. The next identification question is which
source positions and contextual QK products create the clause-order dependence;
more precision tuning cannot answer it.
