from pathlib import Path
import sys,json,torch,importlib.util,hashlib
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal');sys.path.insert(0,str(p))
import typed_face_mlp8_coupled_v1 as coupled
spec=importlib.util.spec_from_file_location('head8',p/'extracted_circuits/odd_attention8h2_typed_face_v1/native.py');head8=importlib.util.module_from_spec(spec);spec.loader.exec_module(head8)
binding=json.loads((p/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu');torch.set_num_threads(2);torch.manual_seed(17092250)
program={'head8':torch.load(p/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True),'mlp8':{a:sd['transformer.h.8.mlp.'+b+'.weight'] for a,b in [('left','Left'),('right','Right'),('down','Down')]}}
g=torch.randn(1,24,1152);current=torch.randn_like(g);donor=torch.randn(1,1152);mask=torch.arange(24)>8;mask[-1]=False;rt,dt=program['head8']['token_ids'][:2].tolist();delta=.5*head8.execute(program['head8'],current,donor,rt,dt,8,mask)
L,R,D=[program['mlp8'][k].double() for k in ['left','right','down']];eps=torch.finfo(torch.float32).eps
mlp=lambda z:((z@L.T)*(z@R.T))@D.T/(z.square().mean(-1,keepdim=True)+eps)
actual=delta.double()+mlp(g.double()+delta.double())-mlp(g.double());pred=coupled.execute(head8,program,current,donor,g,rt,dt,8,mask);err=float((pred-actual).norm()/actual.norm());zero=coupled.execute(head8,program,current,donor,g,rt,dt,8,mask,strength=0)
assert err<1e-10 and float(zero.abs().max())==0
r={'pred_a':True,'relative_formula_error':err,'zero_strength_max':float(zero.abs().max()),'external_native_state_arrays':3,'state_scalars_T32':74880,'static_float_scalars':sum(v.numel() for d in program.values() for v in d.values() if v.is_floating_point()),'token_indices':20,'scope':'CPU synthetic-state learned-weight check only. Three declared state inputs, all MLP8 factors retained, no compression or native fixture certificate. Standalone bundle and native replay pending.'}
(p/'TYPED_FACE_MLP8_COUPLED_V1_CPU_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
