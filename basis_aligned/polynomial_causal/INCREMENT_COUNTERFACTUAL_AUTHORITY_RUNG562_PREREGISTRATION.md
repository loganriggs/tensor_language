# Rung 562 preregistration: increment/successor counterfactual authority

## Purpose

The legacy increment result uses forty digit-only numbered lists. It can show that a component moves digit identity,
but it cannot distinguish a reusable `+1` computation from digit copying, number prediction, or one fixed list format.
R562 freezes a larger counterfactual dataset before any new model output is read.

The proposed causal variable is the numeric state together with the operation that maps the observed sequence to its
next value. The main hypothesis is specifically the `+1` operation, not low rank and not generic numeric information.

## Frozen semantic groups and splits

There are 160 content-addressed groups: 64 FIT, 32 SELECT, 32 FINAL_TEST, and 32 OOD. Lexical pools, prompt templates,
and starting-number pools are disjoint across splits. All seven rows derived from a group remain in that group's split.
FINAL_TEST and OOD are not available for model selection.

Each group contains:

1. a coherent three-item `+1` sequence written with digits, paired with a different coherent `+1` state;
2. the corresponding answer-changing pair written with number words;
3. an answer-changing digit-to-number-word pair, with the operation fixed to `+1`;
4. a necessity row in which only the middle number is made inconsistent while the first number, final observed
   number, and registered expected answer stay fixed;
5. a surface rewrite that preserves all numbers, the `+1` relation, and the expected answer;
6. a repeated-number/copy sequence under a surface rewrite, as a non-`+1` numeric control;
7. a coherent step-two sequence under a surface rewrite, as a second non-`+1` numeric control.

## Construction checks

The CPU builder must establish exact GPT-2 token round trips, single-token answer endpoints in their actual leading-
space form, unique oriented prompt pairs, complete seven-family coverage in every group, one split per group, balanced
base/donor orientations, and the declared split counts. It may not import the model, read logits, or make model calls.

## What this does not establish

Passing the construction audit says only that the counterfactual questions are well specified. A later, separately
preregistered capability experiment must show that bilin18 itself performs the answer-changing tasks and remains stable
on the answer-preserving controls. Only then may causal localization test the legacy L8H7/L8H3 and MLP8--14 hypothesis.
The old rank-4 DAS result is retained as legacy evidence and is not promoted by this dataset.
