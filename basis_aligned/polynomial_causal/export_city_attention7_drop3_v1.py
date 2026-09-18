"""Export physically packed fresh-confirmed city-removal generator."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 target=P/'extracted_circuits/city_attention7_drop3_v1';target.mkdir(exist_ok=True)
 files={'attention7.pt':'CITY_ATTENTION7_DROP3_FRESH_V1_PACKED_ATTENTION7.pt','readers.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt','head8.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt','attention7.py':'city_attention7_packed_v1.py','readers.py':'city_mlp7_readers_v1.py'}
 for name,source in files.items():shutil.copyfile(P/source,target/name)
 code=(P/'city_attention7_drop3_v1.py').read_text().replace('from city_attention7_packed_v1 import','from attention7 import').replace('from city_mlp7_readers_v1 import','from readers import')
 code=code.replace('    generated=attention7',"    if residual6.shape[0]!=1:raise ValueError('Only batch size one is certified')\n    generated=attention7")
 (target/'execute.py').write_text(code)
 r=json.loads((P/'CITY_ATTENTION7_DROP3_FRESH_V1_RESULT.json').read_text())
 isolated=P/'CITY_ATTENTION7_DROP3_FRESH_V1_ISOLATED_RESULT.json'
 extraction=json.loads(isolated.read_text())['passes'] if isolated.exists() else None
 manifest={'schema':'regional.city_attention7_drop3.v1','name':target.name,'certified_complete':False,'counterfactual':'Approximate complete head8.2 city removal; attention7 head3 omitted in generated prefix, full native suffix retained. This does not ablate native attention7 head3.','ports':['residual6[1,T,1152]','token_ids[1,T]','city:int','destination[T]:bool'],'external_native_activation_inputs':1,'native_state_scalars_T32':36864,'static_float_scalars':r['floating_scalars'],'weight_bytes':4*r['floating_scalars'],'supported_sequence_tokens':386,'external_computation':'Native blocks0–6 and native MLP8 plus remaining suffix.','four_traits':{'ood_prediction':r['pred_a'] and r['pred_b'],'extraction':extraction,'selective_removal':all(r['pred_'+k] for k in 'acde'),'composition_reuse':None},'property_scope':'Fresh same-corpus documents with fixed regional probes; not pretraining-disjoint or cross-corpus OOD. Independent composition untested for this subset and failed for related full-city partitions. No matched-effect simplicity-null claim.','evidence':{'fresh_cpu':'../../CITY_ATTENTION7_DROP3_FRESH_V1_RESULT.json','packing_replay':'../../CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json','isolated_cpu':'../../CITY_ATTENTION7_DROP3_FRESH_V1_ISOLATED_RESULT.json'},'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(target.iterdir()) if f.suffix in ['.py','.pt']}}
 if (P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_RESULT.json').exists():
  fw=json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_RESULT.json').read_text())
  manifest['fineweb_variant']={'attention7_program':'../../CITY_ATTENTION7_DROP3_FINEWEB_V1_PACKED_ATTENTION7.pt','token_count':392,'static_float_scalars':fw['floating_scalars'],'prediction_passes':fw['pred_a'] and fw['pred_b'],'isolated_replay_passes':json.loads((P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ISOLATED_RESULT.json').read_text())['passes'],'direction_gate_passes':fw['pred_d'],'positive_capable_pairs':'88/102','receipt':'../../CITY_ATTENTION7_DROP3_FINEWEB_V1_RESULT.json'}
  manifest['property_scope']+=' FineWeb table variant predicts native effects and replays standalone but FAILS directional removal88/102<.90; Pile selective-removal pass does not generalize.'
 (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (target/'README.md').write_text('''# Packed eight-head prefix for city removal

Load attention7.pt, readers.pt, head8.pt with torch.load(weights_only=True), keyed by filename stem. Import execute and call execute.execute(program, residual6, token_ids, city, destination).

Only these files and PyTorch are required. CPU float32 residual6[1,T,1152], integer token_ids[1,T], city index, Boolean destination[T]; returns FP64 attention8 removal delta. Batch1 only; unsupported tokens fail. Frozen vocabulary386 tokens.

Attention7 retains all heads except zero-based head3, with physically packed Q1/K1/Q2/K2/value/output maps. Five MLP7 readers and head8 factors are unchanged. Full first-value table retained for downstream inherited values. RMS8 approximated from non-MLP7 sources; MLP7 remains in all numerators. No fitted proxy or quantization.

Price:22,418,566 FP32 values,89,674,264 bytes;36,864 supplied native floats atT32. Saving884,736 values versus the nine-head generator on the same token vocabulary. Native blocks0–6 and MLP8/suffix remain external. Fresh prediction and selective removal pass. Extraction status/evidence is in manifest. Composition and matched-effect simplicity remain unresolved; this is not a complete circuit or token-only model.
''')
 if (P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_RESULT.json').exists():
  with (target/'README.md').open('a') as out:out.write('\nFineWeb variant:392token table,22,432,390FP32 values. Prediction and isolated execution pass, but removal directionFAILS88/102<90%. See manifest fineweb_variant; do not extend Pile selective-removal evidence to this corpus.\n')
if __name__=='__main__':main()
