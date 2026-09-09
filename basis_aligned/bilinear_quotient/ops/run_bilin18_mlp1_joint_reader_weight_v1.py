"""Restore both rank4 MLP1 readers and their full symmetric quadratic functions.

A: hashes/count8, orthogonality1e-4, FP64 abs1e-8 AND relative1e-9,
FP32 native abs1e-3 AND relative1e-5, reader roundtrip and gauge1e-4.
B/C are descriptive completeness checks for function spans and full input spectrum,
not scientific sufficiency bars. Null: no reduced exact linear input support.
Price: eight native forwards, eight1152x1152 FP64 forms, no training or updates.
All native weights and reader adapters remain required; no circuit adoption.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_function_span_recorded pred_c_full_input_spectrum_recorded
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import pooled_response_projector as pooled
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import circuit_fast_screen_producer as producer
import mlp_in_situ_usage_rank_map_probe as loader
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; POLY=ROOT/'basis_aligned/polynomial_causal'
PRIOR=POLY/'BILIN18_MLP1_JOINT_READER_WEIGHT_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP1_JOINT_READER_WEIGHT_V1_RESULT.json'
READERS=POLY/'BILIN18_MLP1_JOINT_READER_WEIGHT_V1_READERS.json'
BINDING=POLY/'BILIN18_MLP1_JOINT_READER_WEIGHT_V1_SOURCES.json'
TASKS=('temporal','iswas')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def agreement(x,y,atol=1e-8,rtol=1e-9):
    delta=x.double()-y.double();a=float(delta.abs().max())
    r=float(delta.norm())/max(float(y.double().norm()),1e-30)
    finite=bool(x.isfinite().all() and y.isfinite().all())
    return {'max_abs':a,'relative_frobenius':r,'passed':finite and a<=atol and r<=rtol}


def main():
    bindings=json.loads(BINDING.read_text())
    observed={path:sha(path) for path in bindings};assert observed==bindings
    dry={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':8,
         'form_shape':[8,1152,1152],'form_bytes':8*1152*1152*8,'model_updates':0,
         'source_manifest_sha256':sha(BINDING)}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps(dry));return
    assert not OUT.exists() and not READERS.exists()
    signal.alarm(600);started=time.perf_counter()
    backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[1].mlp
    assert len(model.transformer.h)==18 and model.config.n_embd==1152 and not model.config.gated
    assert tuple(mlp.Left.weight.shape)==(4608,1152)
    captures=[]
    def capture(_module,args,output):
        captures.append((args[0].detach().reshape(-1,1152)[:17].clone(),
                         output.detach().reshape(-1,1152)[:17].clone(),int(args[0].shape[0])))
    hook=mlp.register_forward_hook(capture)
    with torch.inference_mode():
        try:
            fitted=pooled.fit(backend);task_rows,modes=ladder.fit_full_modes(backend,fitted)
        finally:hook.remove()
        q=fitted['projector']['MLP1']; reader_maps={task:q@modes['MLP1'][task][:,:4] for task in TASKS}
        assert tuple(q.shape)==(1152,8) and all(tuple(c.shape)==(1152,4) for c in reader_maps.values())
        c=torch.cat([reader_maps[t].T for t in TASKS]).double()
        orth={t:float((v.double().T@v.double()-torch.eye(4,device='cuda')).abs().max()) for t,v in reader_maps.items()}
        serialized={'site':'MLP1','tasks':list(TASKS),'physical_readers':{t:v.cpu().tolist() for t,v in reader_maps.items()},
                    'pooled_projector':q.cpu().tolist(),'task_coordinate_modes':{t:modes['MLP1'][t][:,:4].cpu().tolist() for t in TASKS},
                    'authority_sha256':observed,'source_manifest_sha256':sha(BINDING),
                    'row_sha256':{t:hashlib.sha256(json.dumps(task_rows[t],sort_keys=True).encode()).hexdigest() for t in TASKS}}
        atomic_create_json(READERS,serialized)
        restored=json.loads(READERS.read_text())
        roundtrip=all(torch.equal(torch.tensor(restored['physical_readers'][t],dtype=reader_maps[t].dtype,device='cuda'),reader_maps[t]) for t in TASKS)
        # Deterministic orthogonal gauge; no second fitting or changed task subspace.
        gauge=torch.eye(8,device='cuda',dtype=q.dtype).roll(1,0);gauge[:,::2]*=-1
        gauge_audits={}
        for t in TASKS:
            original=reader_maps[t].double();rotated=((q@gauge)@(gauge.T@modes['MLP1'][t][:,:4])).double()
            gauge_audits[t]=agreement(rotated@rotated.T,original@original.T,1e-4,1e-4)
        left=mlp.Left.weight.detach().double();right=mlp.Right.weight.detach().double();down=mlp.Down.weight.detach().double()
        bias=c@mlp.Down_bias.detach().double();coeff=c@down
        forms=torch.empty((8,1152,1152),device='cuda',dtype=torch.float64)
        for k in range(8):
            raw=left.T@(coeff[k,:,None]*right);forms[k]=(raw+raw.T)/2
        def factor(x):return ((x@left.T)*(x@right.T))@coeff.T+bias
        def quadratic(x):return torch.einsum('bi,kij,bj->bk',x,forms,x)+bias
        generator=torch.Generator(device='cpu').manual_seed(60913)
        x=torch.randn(17,1152,generator=generator,dtype=torch.float64).to('cuda')
        eps=torch.finfo(torch.float32).eps
        n=x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
        audits={'random_factor':agreement(quadratic(x),factor(x)),
                'normalized_formula':agreement((quadratic(x)-bias)/(x.square().mean(-1,keepdim=True)+eps)+bias,factor(n))}
        native_audits=[]
        for i,(inputs,outputs,_) in enumerate(captures):
            audits['captured_factor_'+str(i)]=agreement(quadratic(inputs.double()),factor(inputs.double()))
            native_audits.append(agreement(quadratic(inputs.double()),outputs.double()@c.T,1e-3,1e-5))
        print(json.dumps({'stage':'compiled_and_replayed','forwards':len(captures),'audits':audits,'native_audits':native_audits}),flush=True)
        flat=forms.reshape(8,-1);gram=flat@flat.T
        # Invert only the four-dimensional function Grams; record eigenvalues.
        orthonormal=[];function_eigen={}
        for i,t in enumerate(TASKS):
            block=gram[4*i:4*i+4,4*i:4*i+4];e,v=torch.linalg.eigh(block)
            function_eigen[t]=e.cpu().tolist()
            assert bool((e>0).all())
            orthonormal.append(v@torch.diag(e.rsqrt())@v.T)
        cross=orthonormal[0]@gram[:4,4:]@orthonormal[1]
        cosines=torch.linalg.svdvals(cross)
        # Tall-stack SVD avoids squaring conditioning in a support Gram matrix.
        spectrum=torch.linalg.svdvals(forms.reshape(8*1152,1152))
        normalized=spectrum/spectrum[0];support_rank=int((normalized>1e-10).sum())
        hidden_active={t:int((coeff[i*4:i*4+4].square().sum(0)>0).sum()) for i,t in enumerate(TASKS)}
        shared_active=int(((coeff[:4].square().sum(0)>0)&(coeff[4:].square().sum(0)>0)).sum())
        # Rank diagnostics for the native linear producer dictionaries, not a factor-count search.
        producer_spectra={name:torch.linalg.svdvals(w) for name,w in [('Left',left),('Right',right)]}
        structural={'function_gram':gram.cpu().tolist(),'function_gram_eigenvalues':function_eigen,
          'function_principal_cosines':cosines.cpu().tolist(),
          'function_intersection_dimension_at_one_minus_1e_10':int((cosines>=1-1e-10).sum()),
          'input_support_singular_values':spectrum.cpu().tolist(),'input_support_normalized_singular_values':normalized.cpu().tolist(),
          'input_support_numerical_rank_at_1e_10':support_rank,'discarded_space':'none proposed' if support_rank==1152 else 'numerical candidate only; no exact-zero certificate',
          'native_hidden_atoms_active_per_task':hidden_active,'native_hidden_atoms_shared_active':shared_active,
          'native_linear_producer_spectra':{name:(v/v[0]).cpu().tolist() for name,v in producer_spectra.items()},
          'factor_scope':'Active native atoms demonstrate implementation co-use only. Alternative shared factors or literal proportionality not searched; no semantic identity inferred.'}
        a=len(captures)==8 and roundtrip and max(orth.values())<=1e-4 and all(z['passed'] for z in list(audits.values())+native_audits+list(gauge_audits.values()))
        b=bool(cosines.isfinite().all()) and len(cosines)==4
        cc=bool(spectrum.isfinite().all()) and len(spectrum)==1152
        result={'terminal':'invalid' if not(a and b and cc) else 'full_input_support_no_exact_linear_quotient' if support_rank==1152 else 'numerical_support_candidate_uncertified',
          'predictions':{'pred_a_instrument':a,'pred_b_function_span_recorded':b,'pred_c_full_input_spectrum_recorded':cc},
          'audits':audits,'native_fp32_audits':native_audits,'orthogonality':orth,'gauge_audits':gauge_audits,'reader_roundtrip':roundtrip,
          'structural':structural,'projected_bias':bias.cpu().tolist(),'rms_epsilon':eps,
          'authority_sha256':observed,'runner_sha256':sha(RUNNER),'reader_artifact_sha256':sha(READERS),
          'weight_tensor_sha256':{name:hashlib.sha256(w.cpu().contiguous().numpy().tobytes()).hexdigest() for name,w in [('Left',left),('Right',right),('Down',down)]},
          'price':{'model_forwards':len(captures),'sequence_evaluations':sum(z[2] for z in captures),'native_parameters':sum(p.numel() for p in model.parameters()),
                   'reader_adapter_scalars':c.numel(),'derived_form_shape':list(forms.shape),'derived_form_bytes':forms.numel()*forms.element_size(),
                   'model_updates':0,'retained_background':'entire native model','adoption':False},
          'software':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
          'wall_seconds':time.perf_counter()-started}
        atomic_create_json(OUT,result)
        print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))
        print(json.dumps({'principal_cosines':cosines.cpu().tolist(),'input_support_rank':support_rank,'smallest_largest_input_ratio':float(normalized[-1]),'active_hidden_atoms':hidden_active,'shared_active':shared_active}),flush=True)

if __name__=='__main__':main()
