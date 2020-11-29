import pandas as pd
import os
import sys

data_dir = sys.argv[1]
rel_output = pd.read_csv(sys.argv[2], sep=' ', header=None, names=['qid', 'Q0', 'did', 'rank', 'prob', 'run_name'])
relations = pd.read_csv(os.path.join(data_dir, sys.argv[3]),
                                   sep=' ', header=None, names=['qid', '0', 'did', 'label'])
rel_output['labels'] = 0
for index in range(len(relations)):
    qid = relations.loc[index]['qid']
    did = relations.loc[index]['did']
    rel_output.loc[(rel_output['qid']==qid) & (rel_output['did']==did), ['labels']] = 1
    
mrr = 0.0
for qid in rel_output.qid.unique():
    tmp = rel_output[rel_output['qid'] == qid].sort_values(
                'prob', ascending=False).reset_index()
    trues = tmp.index[tmp['labels'] == 1].tolist()
    # if there is no relevant docs for this query
    if not trues:
    # add to total number of qids or not?
        pass
    else:
        first_relevant = trues[0] + 1  # pandas zero-indexing
        mrr += 1.0/first_relevant

mrr /= len(rel_output.qid.unique())
print("Mean reciprocal rank @ 100: ", mrr)