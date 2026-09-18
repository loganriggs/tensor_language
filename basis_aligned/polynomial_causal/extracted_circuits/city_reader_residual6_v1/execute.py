"""City removal from residual6 prefix; native head8 queries and optional city RMS."""
from attention7 import execute as attention7
from citywrite import execute as write

def execute(program,residual6,token_ids,
            city,rotated_queries,destination,input_rms):
    if residual6.shape[0]!=1:raise ValueError('Certified batch size is one')
    attention7_program,reader_program,head8_program=program['attention7'],program['readers'],program['head8']
    generated=attention7(attention7_program,residual6,token_ids,city)
    lookup={int(t):i for i,t in enumerate(attention7_program['token_ids'].tolist())}
    inherited=attention7_program['first_table'][lookup[int(token_ids[0,city])],256:384][None]
    return write(reader_program,head8_program,
                 normalized_mlp7_city=generated['normalized_mlp7_city'],
                 other_city_sources=generated['other_city_sources'],
                 lambda8=attention7_program['lambdas8'][0],rotated_queries=rotated_queries,
                 inherited_value=inherited,city=city,destination=destination,input_rms=input_rms)
