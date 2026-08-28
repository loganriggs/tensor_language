# Preregistration: complete standalone bilin18 tensor-program identity

Date: 2026-08-28

Scope: role-free exact identity only. No corpus role, compression, selection, or semantic
promotion is opened.

## Why

The attention and MLP banks own all 36 component calls, but the facade still owns the
token embedding, 18 residual lambda pairs and x0 sequencing, parameter-free RMSNorms,
unembedding, and output softcap. The complete program must execute after the checkpoint
model object is destroyed before any compressed whole-model factorial is admissible.

## Frozen implementation

The program stores independent copies of:

- the full 50,304 by 1,152 token embedding;
- all 18 by 2 residual lambdas;
- every dense attention and bilinear MLP tensor, including Down biases and rotary state;
- the independent 50,304 by 1,152 unembedding.

Its forward directly evaluates initial RMSNorm, residual mixing, causal squared-bilinear
attention and its first-value bus, MLP products, final RMSNorm, unembedding, and
$30\tanh(z/30)$. No native checkpoint object is an execution argument.

## Acceptance gates

1. Program and native logits are bitwise identical on the frozen synthetic fixture:
   maximum absolute error 0 and equal SHA256.
2. Covered synthetic CE is bitwise identical.
3. A prefix intervention leaves downstream current tokens fixed, has nonzero native
   downstream effect, and the program reproduces the native intervention logits
   bitwise. This is the first mandatory contextual-program gate.
4. Native and program storage pointers are disjoint. After construction and native
   reference evaluation, the checkpoint model is garbage-collected; the program must
   still replay both fixtures identically.
5. Complete stored-value accounting is exactly 545,904,054 values: 430,003,602 in the
   36-component core and 115,900,452 in the shell. Fitted lookup-table values and native
   calls per forward are zero; total token support is true.
6. Source hashes bind the program, runner, tests, facade, both component-bank sources,
   this preregistration, and the passing simultaneous-component parent receipt.
7. Publication is create-only. Sources and tests are committed before the single
   role-free GPU invocation.

Passing licenses exact complete-model ownership only. It does not license compression,
semantic explanation, OOD generalization, editing, or a nonzero strict simplified-model
recovery claim.
