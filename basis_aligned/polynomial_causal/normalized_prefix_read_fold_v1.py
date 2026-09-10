"""Exact real-arithmetic input-RMS/QK-RMS fold for a fixed prefix memory.

This is a conditional operation with explicit context factors, not discovery of
a smaller semantic program. Query norm evaluation retains the query weights.
"""
import json
from pathlib import Path
import numpy as np
import torch


def direct(raw,q1,q2,rotation,keys1,keys2,values,output,eps_input,eps_query):
    u=raw/np.sqrt(np.mean(raw*raw)+eps_input)
    a=q1@u;b=q2@u
    a=rotation@(a/np.sqrt(np.mean(a*a)+eps_query))
    b=rotation@(b/np.sqrt(np.mean(b*b)+eps_query))
    weights=(keys1@a)*(keys2@b)/(q1.shape[0]**2)
    return weights@values@output.T


def compile_context(q1,q2,rotation,keys1,keys2,values,output):
    return {'left':keys1@rotation@q1,'right':keys2@rotation@q2,
            'payload':values@output.T}


def folded(raw,q1,q2,context,eps_input,eps_query):
    input_energy=np.mean(raw*raw)+eps_input
    denominator=np.sqrt(np.mean((q1@raw)**2)+eps_query*input_energy)*np.sqrt(
        np.mean((q2@raw)**2)+eps_query*input_energy)
    weights=(context['left']@raw)*(context['right']@raw)/(denominator*q1.shape[0]**2)
    return weights@context['payload']


def rounded_rotation(position):
    phase=torch.tensor([position,position*.01],dtype=torch.float32)
    c=phase.cos().bfloat16().double().numpy();s=phase.sin().bfloat16().double().numpy()
    return np.block([[np.diag(c),np.diag(s)],[-np.diag(s),np.diag(c)]])


def controls():
    rng=np.random.default_rng(9100927);error=0.;interaction_error=0.;comparisons=0
    for _ in range(32):
        q1=rng.normal(size=(4,7));q2=rng.normal(size=(4,7));output=rng.normal(size=(3,4))
        source=rng.normal(size=(2,7));background=rng.normal(size=(2,7))
        direct_cube=np.zeros((2,2,2,3));folded_cube=np.zeros_like(direct_cube)
        for memory in (0,1):
            rotation=rounded_rotation(6+memory)
            k1=rng.normal(size=(5+memory,4));k2=rng.normal(size=k1.shape);v=rng.normal(size=k1.shape)
            context=compile_context(q1,q2,rotation,k1,k2,v,output)
            assert np.max(np.abs(rotation.T@rotation-np.eye(4)))>1e-4
            for producer in (0,1):
                for recipient in (0,1):
                    raw=background[recipient]+source[producer]
                    a=direct(raw,q1,q2,rotation,k1,k2,v,output,.2,.3)
                    b=folded(raw,q1,q2,context,.2,.3)
                    direct_cube[producer,memory,recipient]=a
                    folded_cube[producer,memory,recipient]=b
                    error=max(error,float(np.max(np.abs(a-b))));comparisons+=1
        # Preserve the actual same-edit memory-by-recipient interaction as well.
        da=direct_cube[1]-direct_cube[0];db=folded_cube[1]-folded_cube[0]
        ia=da[1,1]-da[1,0]-da[0,1]+da[0,0]
        ib=db[1,1]-db[1,0]-db[0,1]+db[0,0]
        interaction_error=max(interaction_error,float(np.max(np.abs(ia-ib))))
    # Dropping the input-energy factor in the epsilon term is not an exact fold.
    raw=np.full(7,3.);q1=np.full((4,7),.1);q2=np.full((4,7),.15)
    rotation=rounded_rotation(6);k1=np.eye(4);k2=np.eye(4);v=np.ones((4,4));output=np.eye(4)
    context=compile_context(q1,q2,rotation,k1,k2,v,output)
    actual=folded(raw,q1,q2,context,.2,.3)
    wrong_den=np.sqrt(np.mean((q1@raw)**2)+.3)*np.sqrt(np.mean((q2@raw)**2)+.3)
    wrong=((context['left']@raw)*(context['right']@raw)/(16*wrong_den))@context['payload']
    omission=float(np.linalg.norm(wrong-actual)/np.linalg.norm(actual))
    assert max(error,interaction_error)<1e-11 and omission>.1
    return {'passed':True,'random_fixtures':32,'source_memory_recipient_comparisons':comparisons,
            'maximum_output_error':error,'maximum_same_edit_interaction_error':interaction_error,
            'epsilon_correction_omission_relative_error':omission,
            'rounded_rotations_not_assumed_orthogonal':True,
            'scope':'FP64 controls of an exact real-arithmetic prefix read with explicit raw source+background and key/value memory. Does not assert deployed-FP32 equivalence or independent semantic extraction. Query matrices remain for norm evaluation; no structural saving claimed.'}


if __name__=='__main__':
    import sys
    result=controls()
    with Path(sys.argv[1]).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
