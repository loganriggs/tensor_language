# Shared-output regional producers: head8.2 and head9.8

A torch-only executable pair of scalar value computations with their complete native joint QK routing. The two producers use different contextual inputs and write into one shared direction in the regional branch's four-reading interface. **This package requires native layer8/9 normalized attention inputs.** It does not generate context from token IDs alone.

```python
import torch
from execute import execute

program = torch.load("program.pt", map_location=device, weights_only=True)
# current8/current9: FP32 [batch, sequence, 1152], on device
# token_ids: integer [batch, sequence], on device
contribution = execute(current8, current9, token_ids, program)
# FP64 [batch, sequence, 4]
only_first = execute(current8, current9, token_ids, program, masks=(1, 0))
```

The output is the producer contribution to four readings of the residual **before attention17 normalization**; residual propagation coefficients are already folded into the output coefficients. Divide a donor-minus-recipient contribution by the recipient residual17 RMS before adding it to the regional branch's four normalized source readings. The downstream first-token reading, other source contributions, query, and key norms remain recipient-native in the validated intervention.

For producer i, the runtime computes a causal sum of its two normalized QK scores multiplied together and a scalar value. That value is a current-state reading plus a complete-vocabulary lookup of its first-state reading. Both scalar sums are combined through the shared output direction. Native rotary-table rounding and FP32 RMS epsilon are retained.

Price: **1,282,566 scalars; 5,541,936 tensor bytes**, including1,179,648 QK scalars and50,304x2 FP64 lookup entries. The common output itself saves only two scalars versus separate four-coordinate writers; the main structural fact is the common consumer interface. The native input generators and downstream suffix are not included in that byte count.

Validation on48new-city/new-spelling/new-template contexts:

- Original components match their full heads' conditional cue effects within2.3–3.0%; jointly retain89–93%of the selected attention8/9/13 group's cue transfer.
- Shared output changes the joint component effect by.23–.25%.
- This executor matches the frozen shared-output producer contributions within3.02e-7 per head and8.75e-8 jointly.
- Individual/joint swap and removal effects replay within6.86e-5 relative under the registered1e-4bar.

These results support conditional local prediction, execution and grouping. They do not establish full corpus OOD, independent producer-input extraction or preservation of head8.2's known newline-setter service. The newline endpoint on the spelling panel is descriptive; a dedicated capability control remains required.

[Primary mathematics and evidence](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md) · [native execution](../../SCALAR_PRODUCERS_NATIVE_V1_RESULT.json) · [effect replay](../../SCALAR_PRODUCERS_EFFECT_REPLAY_V1_RESULT.json). The manifest binds the program and runtime hashes.
