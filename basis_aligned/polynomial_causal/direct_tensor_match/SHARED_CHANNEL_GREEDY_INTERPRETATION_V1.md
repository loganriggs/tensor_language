# Residual-aware selection of shared native quadratic channels

22 September 2026, 03:16 UTC. This baseline keeps a subset of the native last-MLP products, each shared across the 16 selected output readers, and refits their output coefficients exactly in coefficient space. It leaves input factors fixed. The older native-channel baseline used the full output frame; this selected-output test adds residual-aware greedy selection.

The joint residual selector chooses the atom giving the largest exact orthogonal-projection energy improvement. A same-budget comparator chooses channels by individual contribution energy before the same exact readout refit. Greedy selection is not globally optimal over channel subsets.

Both registered 20% relative-improvement predictions fail. At 512 products, natural weighting gives 68.56% centered error versus 68.70% for individual-energy selection. Equal-output weighting gives 75.26% equal-output error versus 75.38%. The improvement is about 0.2%, not 20%.

| Weighting and selection | Products | Natural centered error | Equal-output centered error |
| --- | ---: | ---: | ---: |
| natural, greedy | 128 | 83.06% | 88.01% |
| natural, greedy | 512 | 68.56% | 76.89% |
| natural, individual_energy | 128 | 83.41% | 88.20% |
| natural, individual_energy | 512 | 68.70% | 76.98% |
| equal_output, greedy | 128 | 84.48% | 86.81% |
| equal_output, greedy | 512 | 70.99% | 75.26% |
| equal_output, individual_energy | 128 | 85.05% | 87.05% |
| equal_output, individual_energy | 512 | 71.74% | 75.38% |

At 512 products the program stores 1,187,840 reader/readout floats, versus the native 4,608 products and 10,690,560 floats. Errors remain too large to adopt this compression. These are centered isotropic-Gaussian/coefficient errors; text-state and intervention fidelity are untested.

Ten dense least-squares comparisons across five planted families pass, including duplicated/cancelling atoms. Native explicit residual contraction agrees with projection energy within 8.1e-16 relative target energy. Native runtime was approximately 4.3 seconds on two CPU threads. Control evidence supports interpreting this negative result, rather than attributing it to an observed projection bug.

The next constructive comparison changes the output basis and factors quadratic slices there, allowing each computed scalar to write to several original outputs. This changes the representation rather than continuing to adjust selection scores on a fixed dictionary.

[Plan](SHARED_CHANNEL_GREEDY_PLAN_V1.md) · [Native results](SHARED_CHANNEL_GREEDY_V1.json) · [Dense controls](SHARED_CHANNEL_GREEDY_CONTROLS_V1.json) · [Selector](shared_channel_greedy.py) · [Native executor](audit_shared_channel_greedy.py).
