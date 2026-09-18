"""Materialize the unfitted single-input program, without repository imports."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    target=P/'extracted_circuits/city_residual6_single_input_v1';target.mkdir(exist_ok=True)
    files={'attention7.pt':'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt','readers.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt','head8.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt','attention7.py':'city_attention7_full_v1.py','readers.py':'city_mlp7_readers_v1.py'}
    for name,source in files.items():shutil.copyfile(P/source,target/name)
    code=(P/'city_residual6_single_input_v1.py').read_text().replace('from city_attention7_full_v1 import','from attention7 import').replace('from city_mlp7_readers_v1 import','from readers import')
    code=code.replace("    generated=attention7", "    if residual6.shape[0]!=1:raise ValueError('Only batch size one is certified')\n    generated=attention7")
    (target/'execute.py').write_text(code)
    cert=json.loads((P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_CPU_RESULT.json').read_text())
    manifest={'schema':'regional.city_residual6_single_input.v1','name':target.name,'certified_complete':False,
      'counterfactual':'Complete head8.2 city-value removal at attention8 destination positions, approximating mixed8 RMS with other-source norm.',
      'ports':['residual6[1,T,1152]','token_ids[1,T]','city:int','destination[T]:bool'],
      'external_native_activation_inputs':1,'native_vector_state_arrays':1,'native_scalar_state_arrays':0,
      'native_state_scalars_T32':36864,'static_float_scalars':cert['floating_scalars'],'weight_bytes':4*cert['floating_scalars'],
      'supported_sequence_tokens':396,'batch_contract':'B=1; CPU float32 inputs; output float64',
      'external_computation':'Native blocks0–6 and full native MLP8/suffix; this program only generates the attention8 edit.',
      'four_traits':{'ood_prediction':None,'extraction':None,'selective_removal':None,'composition_reuse':False},
      'evidence':{'local':'../../CITY_RESIDUAL6_SINGLE_INPUT_V1_CPU_RESULT.json','installed_cpu':'../../CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1_RESULT.json','isolated_cpu':'../../CITY_RESIDUAL6_SINGLE_INPUT_V1_ISOLATED_RESULT.json'},
      'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(target.iterdir()) if f.suffix in ['.py','.pt']}}
    installed=json.loads((P/'CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1_RESULT.json').read_text())
    isolated=json.loads((P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_ISOLATED_RESULT.json').read_text())
    manifest['four_traits']['extraction']=isolated['passes'] and installed['pred_b']
    manifest['property_scope']='Extraction and selective removal pass on opened CPU data; fresh prediction pending; composition failed. Null trait values do not borrow evidence from older operators.'
    manifest['opened_selective_removal']=all(installed['pred_'+k] for k in 'abcde')
    manifest['serialized_program_bytes']=sum(f.stat().st_size for f in target.glob('*.pt'))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'README.md').write_text('''# One residual6 input: complete city-removal write

Load attention7.pt, readers.pt and head8.pt with torch.load(weights_only=True)
into a dictionary keyed by their stems. From this directory, import execute and call
`execute.execute(program, residual6, token_ids, city, destination)`.

Only PyTorch and these files are required. Certified interface: CPU float32
residual6[1,T,1152], integer token_ids[1,T], city index, Boolean destination[T].
The result is a float64 attention8 removal delta. Unsupported tokens and batches
larger than one fail. The frozen token vocabulary has 396 entries.

Five MLP7 readers supply both queries, both keys and current value. Attention7
is generated with all nine heads. Mixed8 RMS is approximated from non-MLP7 sources;
MLP7 remains fully present in each numerator. No supplied queries or RMS scalars.
Native blocks0–6 and MLP8 plus later layers remain external.

Price: 23,326,342 FP32 values (93,305,368 bytes), plus 36,864 native input floats
at T32. Both weight and state counts increase over the earlier three-input export.
This is not token-only extraction or a compression win. Fresh prediction remains
untested for this version; independent source composition previously failed.
Consult manifest evidence links for local, installed and isolated test receipts.
''')
if __name__=='__main__':main()
