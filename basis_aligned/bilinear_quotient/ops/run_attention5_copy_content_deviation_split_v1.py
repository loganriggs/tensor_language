#!/usr/bin/env python3
# BQGATE:49documentforwards;terminal-copy positive/matched cells;180seconds;fixed head groups.
"""Task-defined copy/content split of attention5 mean deviations."""
import json, os, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; P=ROOT/'basis_aligned/polynomial_causal'; BQ=ROOT/'basis_aligned/bilinear_quotient'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(BQ),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
STEM='ATTENTION5_COPY_CONTENT_DEVIATION_SPLIT_V1'; D,V,T,NH,HD,CHUNK=1152,50304,257,9,128,8


@torch.no_grad()
def doc_forward(model,idx):
    x=F.rms_norm(model.transformer.wte(idx),(D,)); x0,first=x,None
    for block in model.transformer.h: x,first=block(x,first,x0)
    return 30.0*torch.tanh(model.lm_head(F.rms_norm(x,(D,)))/30.0)


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']; assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/(STEM+'_CPU_CONTROL.json')).read_text()); assert control['pred_a']
    saved=torch.load(P/'ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt',map_location='cpu',weights_only=True)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('49documentforwards;copy-positive+matched-negative;fixed h7,h5,h6+3,h5+7,h6+7+3,all'); return
    out,artifact=P/(STEM+'_RESULT.json'),P/(STEM+'_ARTIFACT.pt'); assert not out.exists() and not artifact.exists()
    signal.alarm(180); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); mean_z=saved['mean_head_input'].cuda().float(); context={'mode':'native','heads':()}
    cproj=model.transformer.h[5].attn.c_proj
    def hook(module,args,output):
        if context['mode']=='native': return output
        z=args[0].view(args[0].shape[0],args[0].shape[1],NH,HD); w=module.weight.float(); bias=None if module.bias is None else module.bias.float()
        changed=F.linear(mean_z.flatten(),w,bias).view(1,1,D).expand(z.shape[0],z.shape[1],D).clone()
        for h in context['heads']: changed+=F.linear(z[:,:,h].float()-mean_z[h],w[:,h*HD:(h+1)*HD],None)
        return changed.to(output.dtype)
    handle=cproj.register_forward_hook(hook); forwards=0
    def losses(docs,mode,heads=()):
        nonlocal forwards
        context['mode'],context['heads']=mode,tuple(heads); pieces=[]
        for start in range(0,len(docs),CHUNK):
            batch=docs[start:start+CHUNK]; idx,target=batch[:,:T-1].cuda(),batch[:,1:T].cuda(); logits=doc_forward(model,idx); forwards+=1
            pieces.append(F.cross_entropy(logits.reshape(-1,V),target.reshape(-1),reduction='none').view(batch.shape[0],-1).cpu())
        return torch.cat(pieces).double()
    start_time=time.perf_counter(); results={}; saved_losses={}
    arms={'native':('native',()),'mean':('mean',()),'h7':('subset',(7,)),'h5':('subset',(5,)),'h63':('subset',(6,3)),
          'h57':('subset',(5,7)),'h673':('subset',(6,7,3)),'all':('all',range(NH))}
    for label,filename in (('final_natural','final_natural.pt'),('ood_code','ood_code.pt')):
        obj=torch.load(BQ/'.rowcache_terminal_copy_induction_v2'/filename,map_location='cpu',weights_only=False); docs=obj['rows'][:24].long()
        masks={'positive':obj['copy_cells']['positive'][:24],'matched_negative':obj['copy_cells']['matched_negative'][:24]}
        vals={name:losses(docs,*spec) for name,spec in arms.items()}
        cell={}
        for cname,mask in masks.items():
            mean_damage=float((vals['mean'][mask]-vals['native'][mask]).mean())
            recoveries={name:float((mean_damage-float((vals[name][mask]-vals['native'][mask]).mean()))/mean_damage) for name in ('h7','h5','h63','h57','h673')}
            cell[cname]={'count':int(mask.sum()),'mean_damage':mean_damage,'recoveries':recoveries,
                         'all_restore_ce_error':float((vals['all'][mask]-vals['native'][mask]).mean())}
        results[label]=cell; saved_losses[label]=vals
    context['mode']='native'; sample=torch.load(BQ/'.rowcache_terminal_copy_induction_v2/final_natural.pt',map_location='cpu',weights_only=False)['rows'][:CHUNK]
    idx,target=sample[:,:T-1].cuda(),sample[:,1:T].cuda(); manual=float(F.cross_entropy(doc_forward(model,idx).reshape(-1,V),target.reshape(-1))); forwards+=1
    module=float(model(idx.contiguous(),target.contiguous())); handle.remove()
    result={'pred_a':abs(manual-module)<=.01,'pred_b':True,'pred_c':True,'pred_d':True,'pred_e':True,
            'manual_module_ce_error':abs(manual-module),'corpora':results,'body_forwards':forwards,'seconds':time.perf_counter()-start_time,
            'source_shas':binding,'scope':'Task-defined terminal-copy versus matched-negative endpoint split of fixed attention5 deviation head groups.'}
    for cells in results.values():
        pos,neg=cells['positive'],cells['matched_negative']; result['pred_a'] &= abs(pos['all_restore_ce_error'])<=.005 and abs(neg['all_restore_ce_error'])<=.005
        result['pred_b'] &= pos['mean_damage']>=.05
        result['pred_c'] &= pos['recoveries']['h57']>=.50 and pos['recoveries']['h57']-neg['recoveries']['h57']>=.15
        result['pred_d'] &= neg['recoveries']['h673']>=.50 and neg['recoveries']['h673']-pos['recoveries']['h673']>=.15
        result['pred_e'] &= pos['recoveries']['h7']>=.20 and neg['recoveries']['h7']>=.20
    torch.save(saved_losses,artifact); result['artifact_sha256']=digest(artifact); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
