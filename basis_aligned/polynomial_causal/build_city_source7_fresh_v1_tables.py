"""Reuse the weight-only table builder without altering the frozen generator."""
import build_city_residual6_single_input_fresh_v1_tables as builder
if __name__=='__main__':
    builder.STEM='CITY_SOURCE7_FRESH_V1'
    builder.main()
