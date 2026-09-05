import circuit_fast_screen_candidate_task14_fixed_reader_transfer as authority
import run_task14_fixed_reader_transfer_native_capability as run

def test_authority_is_frozen_new_and_balanced():
 rows=authority.build_rows(); assert len(rows)==32 and authority.canonical_sha256(rows)==authority.EXPECTED_AUTHORITY_SHA256
 assert {x["template_id"] for x in rows}=={"inside_above","above_inside"}
def test_gate_has_twelve_eight_row_cells():
 g=run.build_gate(); assert len(g.cells)==12 and {x.expected_count for x in g.cells}=={8}
def test_plan_is_native_only():
 p=run.compile_plan(); assert p["native_only"] and p["endpoint_evaluations"]==96
