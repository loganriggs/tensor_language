# Shared-source continuation program

This package implements a candidate residual write using a shared16-dimensional input dictionary for two quadratic source readers. It stores23,588 scalars and takes native normalized MLP16 input `z` and last-MLP input `h`. It needs no original model weights, QR matrix or calibration cache **after those inputs have been supplied**.

```python
import torch
from source_interface import residual_write
program = torch.load("program.pt", weights_only=True)
# z and h have shape [..., 1152]. Match their device/dtype to the program.
write = residual_write(z.double(), h.double(), program)
```

The executor is [source_interface.py](../../source_interface.py). The output is the contribution to subtract for the tested removal; it is not the full model output. `manifest.json` records literal storage and remaining dependencies.

Frozen source-interchange results meet the registered point-estimate comparison with the unshared41,508-scalar baseline. All-cohort native effect fidelity and signed behavioral fidelity are different: previous ordinary-state continuation CE failure remains unresolved. This is an interface extraction candidate with internal reuse, not a fully isolated or adopted semantic circuit.

**Robustness update:** a balanced-donor follow-up fails the relative preservation tolerance at code spaced-word sites. The initial pass does not establish control-family robustness; this package remains a candidate.
