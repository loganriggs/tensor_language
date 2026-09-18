# MLP8-mediated current-value correction

Load program.pt with torch.load(weights_only=True). Import execute from this folder and call `execute.execute(program,z,delta,edited_rho9)`.

CPU batch1 interface: z and delta[1,T,1152] are native unnormalized MLP8 input and the attention8 edit; edited_rho9[1,T,1] is the edited block9 mixed-residual RMS (FP64, including native FP32epsilon). Return FP64[1,T,128]. Subtract this correction from head9.8 value while retaining the upstream cityswap. Both keys and other heads remain untouched; fullsuffix must recompute. This is not native MLP8 ablation.

Only PyTorch and these two files are required. Invalid norm shape, nonpositive/nonfinite norms, non-CPU input and batch>1 fail. All ordered MLP8 cross terms, quadratic and its normalization change remain coupled. Edited RMS9 remains an external native scalar; the program does not infer it from tokens.

Storage:11,206,658FP32 values,44,826,632bytes. AtT32:36,896 native scalars (one z8 array plus32editedRMS values), plus36,864 intervention scalars. This eliminates unused mixed9 arrays and147,456 weight values from the earlier six-piece helper. It does not eliminate the edited-RMS dependency or price the full external model as free.

Fresh mediator prediction, globalpreservation/direction/null gates pass; independent composition and matched-effect simplicity remain unresolved. See manifest for actual isolated/installed status and primary receipts. Earlier FineWeb failures for the original cityremoval/swap remain in the research record.
