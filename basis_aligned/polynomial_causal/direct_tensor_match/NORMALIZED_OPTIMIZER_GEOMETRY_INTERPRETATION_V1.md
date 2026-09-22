# A scale-transfer confound in the native Muon comparison

2026-09-22 04:31 UTC. Bounded implementation/literature follow-up, not the full three-hour review. That remains due at 04:59. This is a descriptive control, not a preregistered native-performance prediction.

**At the native 96-by-1152 factor shape, the selected default-Muon configuration barely changes normalized directions in 250 steps on a simple known-target task.** This is a concrete reason not to interpret a failed native Muon fit as evidence that the model class cannot learn the target.

Our model uses row-normalized factors but optimizes unconstrained raw parameter matrices initialized with standard-normal entries. Raw row norms therefore grow with the square root of the input width. The earlier learning-rate selection used small toy dimensions. Equal step counts and numerically fixed learning rates do not guarantee equal motion in the normalized feature directions.

We inspected the actual installed PyTorch 2.11.0+cu128 implementation. With a matrix of shape $m\times n$, default Muon scales the learning rate by $\sqrt{\max(1,m/n)}$. The alternative `match_rms_adamw` uses $0.2\sqrt{\max(m,n)}$. Its default weight decay is 0.1; the Adam calls use their default zero weight decay. The local Muon implementation also performs the Newton–Schulz update calculation in bfloat16 even when the parameters are float64. These are properties of the specific implementations we called, not a universal definition of either optimizer. [Versioned PyTorch documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html), [original Muon implementation](https://github.com/KellerJordan/Muon).

The diagnostic fits known unit target directions $t_i$ using

$$
\mathcal L(P)=\frac{1}{2m}\sum_i\left\|\frac{P_i}{\|P_i\|}-t_i\right\|^2.
$$

This has an explicit zero-loss witness: each row parallel to its target with positive scale. We compare shapes 4-by-4 and 96-by-1152, two starts, 250 updates, Adam at 0.1 and Muon at 0.01, using the same initial parameters and targets for methods within each shape/start. The RMS-adjusted variant is a diagnostic alternative; it does not retune the queued native experiment. Weight decay is left at each optimizer's actual default, so this is a comparison of invoked configurations, not an isolated test of the update formula.

| Shape | Optimizer configuration | Median rotation from initialization after 250 steps | Median remaining target angle |
| --- | --- | ---: | ---: |
| 4 × 4 | Adam | 54.8–82.4° | below 0.0001° |
| 4 × 4 | Default Muon | 45.1–57.5° | 15.9–26.0° |
| 96 × 1152 | Adam | 89.4–90.1° | about 0.0002° |
| 96 × 1152 | Default Muon | 3.66–3.68° | 85.9–86.6° |
| 96 × 1152 | RMS-adjusted Muon | 25.04–25.05° | 65.5–66.0° |

The row-norm gauge matters despite not changing the represented linear directions: it changes how a fixed parameter-space update rotates them. Weight decay changes those norms over time and can therefore alter effective angular learning rates. The result complements the earlier 250-versus-500-step toy-duration caveat. It does not establish that any particular alternative rate, scaling, normalization of raw parameters or longer schedule will recover the native quartic.

The native and toy problems also differ in number of rows, conditioning, objective normalization, moment geometry and profiled output coefficients. This test changes shape and aspect ratio together and deliberately uses a simpler objective, so it is not a controlled proof that input width alone explains every difference. The original quartic toy successes and failures remain valid for their stated configurations.

**Decision:** keep the queued four-arm native experiment unchanged. Report its result as a comparison of those configurations. Before a follow-up claims that Muon is inferior or that the architecture failed, use a known-capacity quartic control at wide ambient dimension and inspect normalized-direction movement, step counts and explicit optimizer defaults. Do not silently replace the native configuration after seeing its scores.

[All trajectories, defaults, version and source hash](NORMALIZED_OPTIMIZER_GEOMETRY_V1.json) · [Control implementation](audit_normalized_optimizer_geometry.py).
