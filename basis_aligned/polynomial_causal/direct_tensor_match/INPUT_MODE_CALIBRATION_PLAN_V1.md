# Input-mode calibration and stronger radial bound — 2026-09-20 18:12 UTC

For f=(x^T x)^2, the exact input-unfolding Gram is proportional toI. Projection onto anyr-dimensional linearinputspace retains coefficientenergy r(r+2)/[d(d+2)]. Since CP8 depends onatmost32directions, its error is at leastsqrt(1-32*34/[d(d+2)]) when d≥32. This strengthensbutdoesnotinvalidate theprevious pair-unfoldingrank bound. Validate theprojectionformula withrandomdense projectors atd6/8.

Calibrate independentthree-slot Rademacher inputGram probes onthisknownradialtensor atd128, N256/512/1024/4096,8seeds. Reporttrace andfinite-probe ranktail forr4/16/32 againsttheknowntrue tail sqrt(1-r/d). The purpose is todetect finite-sample spectralbias andprevent declaring a certifiednativebound from stable-looking estimates. No optimization/circuitclaim.
