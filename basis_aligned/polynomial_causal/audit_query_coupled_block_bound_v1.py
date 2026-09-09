"""Exact saved-artifact block obstruction and common-square approximation bound."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import torch
import simultaneous_congruence_reference as S

BASE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def trace(a):return sum(a[i][i] for i in range(len(a)))

def block_certificate(gram,forms):
    adj=S.adjugate(S.rational(gram));pivots={};sources=[];examined=0
    units=[[[int(3*i+j==k) for j in range(3)] for i in range(3)] for k in range(9)]
    for index,form in enumerate(forms):
        b=S.multiply(adj,S.rational(form))
        # Same row-major column convention as toy_consumer_commutant_blocks.
        columns=[sum(S.subtract(S.multiply(e,b),S.multiply(b,e)),[]) for e in units]
        for entry,row in enumerate(zip(*columns)):
            row=list(row);examined+=1
            assert row[0]+row[4]+row[8]==0  # identity is in the exact nullspace
            for col,prior in sorted(pivots.items()):
                factor=row[col]
                if factor:row=[x-factor*y for x,y in zip(row,prior)]
            nonzero=next((j for j,x in enumerate(row) if x),None)
            if nonzero is not None:
                scale=row[nonzero];pivots[nonzero]=[x/scale for x in row]
                sources.append({'form':index,'entry':[entry//3,entry%3],'pivot_column':nonzero})
            assert len(pivots)<=8
            if len(pivots)==8:
                return {'scalar_only':True,'rank':8,'nullity':1,'equations_examined':examined,'pivot_sources':sources}
    return {'scalar_only':False,'rank':len(pivots),'nullity':9-len(pivots),'equations_examined':examined,'pivot_sources':sources}

def squared_commutator(gram,a,b):
    adj=S.adjugate(S.rational(gram));u=S.multiply(adj,S.rational(a));v=S.multiply(adj,S.rational(b))
    comm=S.subtract(S.multiply(u,v),S.multiply(v,u))
    denominator=4*trace(S.multiply(u,u))*trace(S.multiply(v,v))
    assert denominator>0
    value=-trace(S.multiply(comm,comm))/denominator
    assert value>=0
    return value

def controls():
    g=[[1,0,0],[0,1,0],[0,0,1]];a=[[1,0,0],[0,2,0],[0,0,3]]
    block=[[1,0,0],[0,1,1],[0,1,0]];dense=[[1,1,0],[1,0,1],[0,1,1]]
    w=[[1,2,0],[0,1,1],[0,0,1]];transform=lambda m:S.multiply(S.multiply(S.transpose(w),m),w)
    checks={'coupled_1_plus_2':block_certificate(g,[a,block])['nullity']==2,
            'dense_pair_irreducible':block_certificate(g,[a,dense])['scalar_only'],
            'diagonal_zero_bound':squared_commutator(g,a,a)==0,
            'congruence_block':block_certificate(transform(g),[transform(a),transform(block)])['nullity']==2,
            'congruence_irreducible':block_certificate(transform(g),[transform(a),transform(dense)])['scalar_only'],
            'congruence_bound':squared_commutator(g,a,dense)==squared_commutator(transform(g),transform(a),transform(dense))}
    return {'passed':all(checks.values()),'checks':checks}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    out=BASE/'QUERY_COUPLED_BLOCK_BOUND_V1_RESULT.json';assert not out.exists()
    source=BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_COEFFICIENTS.pt'
    assert digest(source)=='3f195fe72cc86a62b96ba08733c89e4768a6462608b97a0cbf4483fe1b4e3151'
    parent=json.loads((BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_a_instrument'];check=controls();assert check['passed']
    saved=torch.load(source,map_location='cpu',weights_only=True)
    g=saved['gram'].tolist();forms=saved['forms'].tolist();certificate=block_certificate(g,forms)
    pair=parent['certificate']['witness']['forms'];c2=squared_commutator(g,*(forms[k] for k in pair))
    epsilon=Fraction(1,100);threshold=(2*epsilon+epsilon**2)**2
    result={'scope':'Exact stored dyadic query-interface coefficients only; no ideal-real native interval certificate or structural reduction.',
        'predictions':{'pred_a_instrument':check['passed'],'pred_b_no_proper_common_block':certificate['scalar_only'],
                       'pred_c_one_percent_common_squares_impossible':c2>threshold},
        'block_certificate':certificate,'approximation_bound':{'fixed_form_pair':pair,'commutator_squared_exact':str(c2),
            'one_percent_squared_threshold_exact':str(threshold),'lower_bound_numeric':math.sqrt(1+math.sqrt(float(c2)))-1,
            'metric':'maximum of the two per-form relative Frobenius errors after whitening the fixed native Gram'},
        'controls':check,'source_sha256':digest(source),'runner_sha256':digest(Path(__file__)),
        'reference_sha256':digest(BASE/'simultaneous_congruence_reference.py'),
        'prereg_sha256':digest(BASE/'QUERY_COUPLED_BLOCK_BOUND_V1_PREREGISTRATION.md'),
        'opaque_export_constants':387968,'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
