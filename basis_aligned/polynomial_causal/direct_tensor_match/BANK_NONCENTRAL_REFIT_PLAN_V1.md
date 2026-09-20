# Repair the bank fitting metric, not its rank

The registered coefficient-weighted width8 refactor retained99.77% energy but
failed composition: fresh64error25.86% versus17.44% original. Width12 passes a
useful tradeoff:38144coefficients22products,fresh64error17.95%.

Exact noncentral Gaussian bank audit exposes the metric mismatch: width8 bank
coefficient error4.78% becomes30.49% function error, dominated by mean-induced
constant and linear terms. Width12:1.32% coefficient vs5.08% function error.
Root sensitivity remains a local surrogate, so fixing the bank metric is not
a guarantee of full quartic preservation. A final output constant cannot undo
all errors from wrong internal feature means entering root products.

Use the exact embedding for symmetric Q under x=m+z,z~N(0,I):
E(Q)=[sqrt(2)vec(Q),2Qm,tr(Q)+m^T Qm]. Its Euclidean inner product equals the
Gaussian function inner product. Independent quadrature error5.1e-16.
The helper embedding is implemented in audit_bank_function_metric.py.

Next fit: widths8/12,Adam/Muon, seeds0/1, initialize corresponding previous
optimizer .03 fits;500steps,.005 cosine schedule, normalized unit functional
features and writer penalty.001. Eight arms, select by regularized function
objective only. Keep root and vocabulary frames fixed, Gaussian teacher mean
corrected. Evaluate full quartic and literal graph after selection. Predictions:
width8 reduces Gaussian bank function error by>=50% relative to its starting
program; width8 fresh64error<=.18440681; export price28912coefficients18products
and replay<1e-10. Preserve failure; no automatic additional rate sweep.
