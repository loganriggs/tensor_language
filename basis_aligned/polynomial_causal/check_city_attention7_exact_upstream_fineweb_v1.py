"""Corrected exact-RMS upstream fold on the existing FineWeb fixture panel."""
from pathlib import Path
import hashlib,json,os,time,torch
P=Path(__file__).resolve().parent
from city_attention7_exact_upstream_v1 import execute
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();data=torch.load(P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ARTIFACT.pt',weights_only=True);attention=torch.load(P/'CITY_ATTENTION7_DROP3_FINEWEB_V1_ATTENTION7.pt',weights_only=True);readers=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True);head=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True);binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(ck,weights_only=True,mmap=True);program={'attention7':attention,'readers':readers,'head8':head,'mlp7_left':sd['transformer.h.7.mlp.Left.weight'],'mlp7_right':sd['transformer.h.7.mlp.Right.weight'],'mlp7_down':sd['transformer.h.7.mlp.Down.weight'],'mlp7_bias':sd['transformer.h.7.mlp.Down_bias']};errors=[];outside=[]
    for f in data['fixtures']:
        got=execute(program,f['inputs']['residual6'],f['inputs']['token_ids'],f['inputs']['city'],f['inputs']['destination']);errors.append(float((got-f['native_delta']).norm()/f['native_delta'].norm()));outside.append(float(got[:,~f['inputs']['destination']].abs().max()))
    r={'pred_a':max(errors)<=.05,'pred_b':max(outside)==0 and bool(torch.isfinite(torch.tensor(errors)).all()),'max_relative_write_error':max(errors),'max_off_support':max(outside),'fixtures':len(data['fixtures']),'panel_scope':'Existing fresh FineWeb drop3 fixture panel; exact-fold confirmation is opened relative to row selection, not a new document panel.','scope':'Corrected exact RMS8 closure at residual6-to-attention8 boundary; full suffix and four-property fresh confirmation remain open.','seconds':time.perf_counter()-start,'source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['city_attention7_exact_upstream_v1.py','check_city_attention7_exact_upstream_fineweb_v1.py']}}
    (P/'CITY_ATTENTION7_EXACT_UPSTREAM_FINEWEB_V1_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
