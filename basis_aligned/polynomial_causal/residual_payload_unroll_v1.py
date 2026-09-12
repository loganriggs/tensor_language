"""Exact scalar residual recurrence for pre-attention layer17 payload readers."""
import torch

def coefficients(lambdas):
    # lambdas[0:18,0:2]; updates only through block16, block17 attention excluded.
    depth=len(lambdas)-1;embedding=lambdas.new_tensor(1.)
    for row in lambdas[:depth]:embedding=row[0]*embedding+row[1]
    embedding=lambdas[depth,0]*embedding+lambdas[depth,1]
    scales=lambdas.new_empty(depth);running=lambdas[depth,0]
    for j in range(depth-1,-1,-1):
        scales[j]=running;running=running*lambdas[j,0]
    return embedding,scales

def projected_parts(embedding_reads,attention_reads,mlp_reads,lambdas):
    coefficient,scales=coefficients(lambdas)
    shape=(len(scales),)+(1,)*(attention_reads.ndim-1)
    return coefficient*embedding_reads,scales.reshape(shape)*attention_reads,scales.reshape(shape)*mlp_reads
