# Fixed-writer self-product through MLP10

This is a **port-conditioned computational component**, not a standalone text-to-behavior circuit. `program.pt` stores one FP64 vector of length1152:

$$
c=D_{10}[(L_{10}\lambda w)\odot(R_{10}\lambda w)].
$$

`execute(a,rho,c)` returns $a^2c/\rho$, the direct residual writer's squared contribution to the changed MLP10 output. Supply the removal amplitude and the **actual whole-branch** mean-square input norm plus native FP32 epsilon. The component does not generate those inputs, attention, pristine state or suffix. Its numerator is even in amplitude; its full value need not be even if the supplied normalizer changes with amplitude.

`joint_local` computes the exact signed difference of this component across child, remainder and parent branches. It must not be substituted for the three separate nonlinear suffix evaluations when predicting endpoint interactions.

Native removal of this node from all changed branches alters regional target interactions by8–18%and FineWeb targets by18–33%on the160historical-prefix panel. Controls also change, so selective regional semantics are not established. This is the fixed self-product $P_{00}/2$of the exact five-bank construction; other products containing the writer remain.

The runtime needs only the supplied ports, PyTorch and the stored vector. The full checkpoint is needed to reproduce the extraction audit, not to execute this component. No fresh OOD, independent text-level extraction, or whole-model compression claim is made. See [the method and native removal evidence](../../FIXED_WRITER_PRODUCTS_V1_MATH.md).

Position-consumer testing is now available: final-position removal alone misses the all-position regional target effect by35–42%. Combining final and earlier removals predicts it within0.68–1.42%on regional targets, but fails one FineWeb group's composition criterion. These are historical-panel conditional interventions through the native suffix, not a universal additivity guarantee.
