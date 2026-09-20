"""Shared quadratic-product pruning with explicitly priced sparse supports.

Scores are norms of individual native-space tensor atoms. They are invariant to
diagonal latent rescaling, not general basis rotation, and are not causal scores.
"""
import torch


def pair_scores(block, encoder, output_decoder):
    i,j=torch.triu_indices(block['inputs'],block['inputs'],device=encoder.device)
    gram=encoder.T@encoder
    input_norm=(.5*(gram[i,i]*gram[j,j]+gram[i,j].square())).sqrt()
    output_norm=(output_decoder@block['coefficients']).norm(dim=0)
    return input_norm*output_norm


def prune_pairs(runtime, encoders, decoders, keep, strategy='tensor_norm', seed=668):
    if strategy not in ('tensor_norm','random'):raise ValueError(strategy)
    blocks=[];scores=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    for block,encoder,decoder in zip(runtime['blocks'],encoders,decoders):
        if 'pair_i' in block:raise ValueError('prune from the dense parent')
        score=pair_scores(block,encoder,decoder)
        if not 0<=keep<=len(score):raise ValueError('invalid support budget')
        order=(score.argsort(descending=True) if strategy=='tensor_norm' else
               torch.randperm(len(score),generator=generator).to(score.device))
        selected=order[:keep].sort().values
        i,j=torch.triu_indices(block['inputs'],block['inputs'],device=score.device)
        new=dict(block,coefficients=block['coefficients'][:,selected].clone(),
                 pair_i=i[selected].clone(),pair_j=j[selected].clone())
        blocks.append(new);scores.append(score)
    if len(blocks)!=len(runtime['blocks']):raise ValueError('missing encoder or decoder')
    return dict(runtime,blocks=blocks),scores
