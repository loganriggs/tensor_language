"""City removal from residual6 prefix; native head8 queries and optional city RMS."""
from city_attention7_generator_v1 import execute as attention7
from city_mlp7_generated_norm_v1 import execute as write

def execute(attention7_program,reader_program,head8_program,residual6,token_ids,
            city,rotated_queries,destination,input_rms=None):
    generated=attention7(attention7_program,residual6,token_ids,city)
    lookup={int(t):i for i,t in enumerate(attention7_program['token_ids'].tolist())}
    inherited=attention7_program['first_table'][lookup[int(token_ids[0,city])],256:384][None]
    return write(reader_program,head8_program,
                 normalized_mlp7_city=generated['normalized_mlp7_city'],
                 other_city_sources=generated['other_city_sources'],
                 lambda8=attention7_program['lambdas8'][0],rotated_queries=rotated_queries,
                 inherited_value=inherited,city=city,destination=destination,input_rms=input_rms)
