"""Saved-native probability audit; no model, fit, or new field intervention."""
import hashlib
import json
from pathlib import Path
import torch
import entity_equivariance_reference as E

BASE=Path(__file__).resolve().parent;OUT=BASE/'ENTITY_EQUIVARIANCE_LOWER_BOUND_V1.json'
ROWS=BASE/'GLOBAL_ENTITY_GATE_REUSE_V1_ROWS.pt'


def main():
    torch.set_num_threads(2);checks=E.controls();assert checks['passed'] and not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True);result={};saved={};valid=True
    for pop,block in data.items():
        old=block['query_logits']['native'];new=block['query_logits']['renamed_native'];aligned=torch.empty_like(new)
        for world in range(8):
            ix=[i for i,m in enumerate(block['metadata']) if m['world']==world];first=ix[0]
            mapping=torch.arange(29);mapping[block['tokens'][first,:48:2]]=block['renamed_tokens'][first,:48:2]
            valid &= bool(torch.equal(mapping[:24].sort().values,torch.arange(24)))
            mapped=block['tokens'][ix].clone();entity=mapped<24;mapped[entity]=mapping[mapped[entity]]
            valid &= bool(torch.equal(mapped,block['renamed_tokens'][ix]))
            aligned[ix]=new[ix][:,mapping]
        radius,_=E.information_radius(old,aligned);radius=radius.clamp_min(0);groups={}
        for hop in range(4):
            sel=torch.tensor([m['hop']==hop for m in block['metadata']]);v=radius[sel]
            groups[str(hop)]={'pairs':int(sel.sum()),'minimum_paired_mean_query_kl':float(v.mean()),
                             'pair_radius_p95':float(torch.quantile(v,.95)),
                             'native_argmax_equivariance_rate':float((old.argmax(-1)==aligned.argmax(-1))[sel].double().mean()),
                             'strict_equivariance_not_ruled_out_by_pairwise_bound':bool(v.mean()<=1e-3)}
        mean=float(radius.mean());result[pop]={'pairs':len(old),'minimum_paired_mean_query_kl':mean,
            'all_token_mean_kl_lower_bound_from_queries_only':mean/51,
            'strict_equivariance_ruled_out_at_1e3_query_bar':mean>1e-3,'hop_groups':groups}
        saved[pop]=radius
    receipt={'scope':'exact lower bound on paired original/renamed query distribution fidelity; no model execution or fitting',
             'formula':'0.5 KL(p||r)+0.5 KL(q||r) = JS(p,q)+KL((p+q)/2||r), q=aligned renamed native',
             'controls':checks,'token_alignment_valid':valid,'populations':result,
             'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),
             'reference_sha256':hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
             'primary_reference':'https://www.cise.ufl.edu/~anand/sp06/jensen-shannon.pdf',
             'limits':'pairwise bound need not be attainable by one globally consistent equivariant model; all-token floor uses only final queries; not a bound on nonsymmetric models'}
    assert valid;OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
