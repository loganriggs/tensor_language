"""Fixed-context shared squared-channel test; see immutable V1 preregistration."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(BASE)]
import torch
import exact_source_edit_reference as E
import overlapping_join_normalizer_reference as N
import join_origin_writer_reference as O
import forward_endpoint_program_reference as F
import frozen_payload_lineage_reference as L
import frozen_query_read_reference as R
import key_payload_interaction_reference as I
import query_direction_reference as Q
import field_intervention_metrics as M
import simultaneous_congruence_reference as S

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x):return x-x.mean(-1,keepdim=True)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    out=BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_RESULT.json'
    rows=BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_COEFFICIENTS.pt'
    assert not out.exists() and not rows.exists()
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    checks=S.controls();audits=[];native_audits=[];norm_audits=[]
    with torch.inference_mode():
        w=N.populations()['iid'][0];tok=w['tokens'][3:4];metadata=w['metadata'][3]
        assert metadata['query']==0 and metadata['hop']==3
        context=program.prepare(tok);cache=L.prepare(program.background,tok);embed=cache['embedding'];roots=[]
        for positions in ([49],[48,50],list(range(48))):
            root=torch.zeros_like(embed);root[:,positions]=embed[:,positions]
            roots.append(L.propagate(program.background,root,cache)[0,-1])
        roots=torch.stack(roots);gram=roots@roots.T/roots.shape[-1]
        closure=M.correspondence(roots.sum(0),context['x'][0,-1])
        key=O.origin_write(program.background,tok,{1:w['masks'][1][0]},{1:2})[1]
        mask=w['masks'][0][3:4].clone();mask[:,:,:48:2]=False
        payload=F.messages(program.background,tok,mask)
        def evaluate(z):
            raw=(z@roots)[None];changed=Q.context(program,context,raw)
            interaction=I.interaction(program,changed,key,payload)[:,-1]
            total=R.read(program,changed,payload)
            return torch.stack((center(interaction)[0],center(total-interaction)[0]))
        eye=torch.eye(3,dtype=torch.float64);pure=[evaluate(z) for z in eye]
        forms=torch.zeros(2,29,3,3,dtype=torch.float64)
        for i in range(3):forms[:,:,i,i]=pure[i]
        for i in range(3):
            for j in range(i+1,3):
                forms[:,:,i,j]=forms[:,:,j,i]=(evaluate(eye[i]+eye[j])-pure[i]-pure[j])/2
        native=center(context['logits'][0,-1])
        for amplitudes in ((1.,1.,1.),(.3,-.7,1.2),(0.,0.,0.),(-1.,2.,.5)):
            z=torch.tensor(amplitudes,dtype=torch.float64);raw=(z@roots)[None]
            predicted=torch.einsum('i,roij,j->ro',z,forms,z);explicit=evaluate(z)
            audits.append(M.correspondence(predicted+native,explicit+native))
            norm_audits.append(M.correspondence(z@gram@z+gram.trace(),raw.square().mean()+gram.trace()))
            with Q.native_hooks(model,context['x'],raw):arms=I.native_arms(model,context['x'],key,payload)
            a,b,c,d=(center(arms[k][0,-1]) for k in ((False,False),(True,False),(False,True),(True,True)))
            native_audits.extend((M.correspondence(predicted[0]+b+c,a+d),M.correspondence(predicted[1]+d,b)))
        tensor_payload={'gram':gram,'forms':forms.reshape(58,3,3),'roots':roots,'tokens':tok,
                        'metadata':metadata,'root_order':['L','H','D'],'form_order':['I:'+str(i) for i in range(29)]+['J:'+str(i) for i in range(29)]}
        assert max(v.numel()*v.element_size() for v in tensor_payload.values() if isinstance(v,torch.Tensor))<256*1024**2
        torch.save(tensor_payload,rows)
        stored=torch.load(rows,map_location='cpu',weights_only=True)
        try:
            certificate=S.obstruction(stored['gram'].tolist(),stored['forms'].tolist());positive_gram=True
        except ValueError as exc:
            certificate={'assumption_failure':str(exc)};positive_gram=False
        diagnostic=None
        if positive_gram and certificate['witness'] is not None:
            i,j=certificate['witness']['forms'];p=torch.linalg.inv(torch.linalg.cholesky(gram)).T
            aw,bw=(p.T@stored['forms'][k]@p for k in (i,j))
            diagnostic=float(torch.linalg.matrix_norm(aw@bw-bw@aw)/(2*torch.linalg.matrix_norm(aw)*torch.linalg.matrix_norm(bw)))
    instrument=checks['passed'] and positive_gram and all(a['passed'] for a in audits+native_audits+norm_audits+[closure])
    result={'scope':'Exact dyadic obstruction for compiled forms in one fixed opened context; numerical native correspondence, no ideal-real interval certificate.',
        'predictions':{'pred_a_instrument':instrument,'pred_b_common_squared_channels':certificate.get('simultaneously_diagonalizable') if instrument else None},
        'certificate':certificate,'whitened_normalized_commutator_diagnostic':diagnostic,
        'compiler_max_abs':max(a['max_abs'] for a in audits),'native_max_abs':max(a['max_abs'] for a in native_audits),
        'norm_max_abs':max(a['max_abs'] for a in norm_audits),'lineage_max_abs':closure['max_abs'],
        'controls':checks,'metadata':metadata,'opaque_export_constants':program.independent_constant_count(),
        'coefficients_sha256':digest(rows),'package_sha256':digest(package),
        'source_sha256':{name:digest(BASE/(name+'.py')) for name in ('query_direction_reference','simultaneous_congruence_reference','key_payload_interaction_reference','frozen_payload_lineage_reference','overlapping_join_normalizer_reference','join_origin_writer_reference','forward_endpoint_program_reference','frozen_query_read_reference')},
        'prereg_sha256':digest(BASE/'QUERY_COMMON_SQUARED_CHANNELS_V1_PREREGISTRATION.md'),
        'runner_sha256':digest(Path(__file__)),'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
