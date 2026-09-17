"""Conditional two-stage write, with three explicit native-state arrays.

Computational composition is not evidence for small causal interactions.
Head8 and head9 weight programs are explicit arguments; no model import.
"""
from extracted_circuits.odd_attention8h2_typed_face_v1 import native as head8
from extracted_circuits.odd_value_delta_raw_v1.execute import Program as Head9


class Program:
    def __init__(self,head8_weights,head9_weights,lambda90):
        self.head8=head8_weights
        self.head9=Head9(head9_weights)
        self.lambda90=lambda90

    def execute(self,current8,donor_city8,raw_mixed9,recipient_token,donor_token,city,destination,strength=.5):
        delta8=strength*head8.execute(self.head8,current8,donor_city8,recipient_token,donor_token,city,destination)
        return self.head9.execute(raw_mixed9,delta8,self.lambda90,destination)
