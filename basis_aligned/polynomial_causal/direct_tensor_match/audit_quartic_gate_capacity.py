"""Pool fixed first-slot sketches and apply proven input-span gate bounds."""
import json,math
from pathlib import Path
import torch
from quartic_reader_rank_bound import analyze
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);source=torch.load(P/'QUARTIC_SLOT_CAPACITY_GRAMS_V1.pt',weights_only=True);grams=source['grams'];result,_=analyze([sum(grams)/len(grams)],[256,512,768,1024]);pooled=result['panels'][0];need=pooled['necessary_reader_counts']['0.1'];k=math.ceil(need/64)
 out=dict(pooled=pooled,ten_percent_necessary_general_scalar_products=math.ceil(need/2),ten_percent_necessary_products_per_quadratic_feature_at_width32=k,necessary_architecture_product_budget_if_same_pair_compiler=32*k+256,scope='Exactnecessityforpooled512-triplefirst-slotmetric, not certifiedfullFrobeniusnorm. Generaldivision-free scalararithmeticDAG homogeneousquartic with M variablemultiplications hasinputspanrank<=2M. Current32featuresxkproducts needsinputspan<=64k. Counts necessarynotsufficient. No data/circuitadoption.')
 (P/'QUARTIC_GATE_CAPACITY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
