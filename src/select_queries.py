import pandas as pd
import numpy
import os
import sys
data_dir = sys.argv[1]
top100 = pd.read_csv(os.path.join(data_dir, f'msmarco-doctrain-top100'),
                                   sep=' ', header=None, names=['qid', 'Q0', 'did', 'rank', 'score', 'run'])
queries_id = top100.qid.unique()
queries_id = numpy.array(queries_id)
selected_queries = numpy.random.choice(queries_id, 50000, replace=False) 
with open(os.path.join(data_dir, "msmarco-doctrain-selected-queries.txt"), "w+") as file:
    for i in range(len(selected_queries)):
        file.write(str(selected_queries[i]) + "\n")
