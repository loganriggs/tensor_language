from pathlib import Path
import json,torch
from joint_attention_mixed_ports_v1 import decompose
from joint_attention_three_group_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'CROSSFIRST_ATTENTION17_PORTS_V1_ARTIFACT.pt',weights_only=True);bind=json.loads((P/'CROSSFIRST_ATTENTION17_PORTS_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in bind if k.endswith('pytorch_model.bin'));sd=torch.load(checkpoint,weights_only=True,mmap=True);W=sd['transformer.h.17.attn.c_proj.weight'][:,256:384].double();h=torch.load(P/'CROSSFIRST_READOUT_SPLIT_V1_ARTIFACT.pt',weights_only=True)['final_states'].double();bar=h[:,1]+h[:,3]-h[:,0];states=[];errs=[];omitted=[]
 for i,ports in enumerate(a['ports']):
  args=[ports[k] for k in (0,1,3,4)];d=decompose(*args);selected=sum(v for k,v in d['cross_terms'].items() if k[-1]!='0')+d['defect_terms']['1']+d['defect_terms']['2'];small=execute(*args);errs.append(float((small-selected).norm()/selected.norm().clamp_min(1e-30)));full=d['total'];omitted.append(float((small-full).norm()/full.norm().clamp_min(1e-30)));states.append(torch.stack([bar[i],bar[i]+(full@W.T).flatten(),bar[i]+(small@W.T).flatten()]))
 assert max(errs)<1e-10
 torch.save(dict(states=torch.stack(states).float()),P/'CROSSFIRST_THREE_GROUP_V1_INPUT.pt');result=dict(max_selected_identity_error=max(errs),regional_max_write_error=max(omitted[:96]),FineWeb_max_write_error=max(omitted[96:]),source_contractions=3,scope='Exact regrouping of selected19-term-expansion groups oncachednativeports; reducedprogramomits terms, nativeeffect screen pending. No fitting or freshvalidation; fourportcorners/nativebackground remain.')
 (P/'CROSSFIRST_THREE_GROUP_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
