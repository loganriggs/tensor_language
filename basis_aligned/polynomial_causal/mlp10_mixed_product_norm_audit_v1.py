"""Native outcome audit of the direct MLP10 product, with distinct targets."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent


def main():
    a=torch.load(P/'MLP10_MIXED_PRODUCT_NORM_V1_ARTIFACT.pt',weights_only=True)['measures'].double();cells=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
        z=a[lo:hi,:,0];mlp=z[:,6]-z[:,5];whole=z[:,4]-z[:,5];cross=z[:,8]-z[:,5];norm=z[:,9]-z[:,5];formula=z[:,10]-z[:,5]
        cells.append(dict(cell=label,cross_mlp_same_nonzero_sign=int((cross*mlp>0).sum()),cross_mlp_opposite_sign=int((cross*mlp<0).sum()),mlp_reference_zeros=int((mlp==0).sum()),
            cross_whole_block_error=float((cross-whole).norm()/whole.norm()),cross_whole_block_same_sign=int((cross*whole>0).sum()),
            meanabs_cross_effect=float(cross.abs().mean()),maxabs_cross_mlp_error=float((cross-mlp).abs().max()),
            maxabs_formula_mlp_error=float((formula-mlp).abs().max()),normalization_norm_over_mlp=float(norm.norm()/mlp.norm()),
            normalization_mlp_cosine=float((norm*mlp).sum()/(norm.norm()*mlp.norm()).clamp_min(1e-30))))
    result=dict(cells=cells,scope='Existing native intervention outcomes. Direct product retains the joint RMS denominator; small normalization remainder doesnot license dropping allnormalization. FineWeb absolute errors andzeros remain reported.')
    (P/'MLP10_MIXED_PRODUCT_NORM_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
