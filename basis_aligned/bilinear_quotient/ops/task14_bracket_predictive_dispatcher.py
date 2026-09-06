#!/usr/bin/env python3
"""Pure API for the compiled predictive/manipulable two-program package."""
from __future__ import annotations
from typing import Mapping

WIDTH=1152;NUMBERS=("singular","plural");CLOSERS=(1,8,60)
class DispatchError(ValueError):pass
def dispatch_task14(package:Mapping[str,object],*,recipient_number:str,donor_number:str,cardinality:int)->dict:
 if recipient_number not in NUMBERS or donor_number not in NUMBERS or recipient_number==donor_number:raise DispatchError("Task14 requires two licensed opposite numbers")
 if type(cardinality) is not int or cardinality not in range(5):raise DispatchError("cardinality must be 0 through 4")
 key=f"{recipient_number}_to_{donor_number}.cardinality_{cardinality}"
 try:item=package["programs"]["task14"][key] # type: ignore[index]
 except (KeyError,TypeError) as error:raise DispatchError(f"missing Task14 entry {key}") from error
 if len(item["displacement"])!=WIDTH:raise DispatchError("bad Task14 width")
 return {"operation":"add_displacement","site":"layer11.head3.final_position","key":key,"vector":item["displacement"],"predicted_donorward_effect":item["predicted_donorward_effect"]}
def dispatch_bracket(package:Mapping[str,object],*,recipient_closer_id:int,donor_closer_id:int)->dict:
 if recipient_closer_id not in CLOSERS or donor_closer_id not in CLOSERS:raise DispatchError("closer ID outside licensed vocabulary")
 if recipient_closer_id==donor_closer_id:return {"operation":"no_edit","site":"layer13.head8.semantic_opener_position","key":f"{recipient_closer_id}->{donor_closer_id}","vector":None,"predicted_donorward_effect":0.0}
 pair=f"{recipient_closer_id}->{donor_closer_id}"
 try:term=package["programs"]["bracket"]["absolute_terms"][str(donor_closer_id)];effect=package["programs"]["bracket"]["predicted_effects"][pair] # type: ignore[index]
 except (KeyError,TypeError) as error:raise DispatchError(f"missing bracket entry {pair}") from error
 if len(term)!=WIDTH:raise DispatchError("bad bracket width")
 return {"operation":"replace_absolute","site":"layer13.head8.semantic_opener_position","key":pair,"vector":term,"predicted_donorward_effect":effect}

