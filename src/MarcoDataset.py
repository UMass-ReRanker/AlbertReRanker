import numpy 
import os
import pandas as pd
import string
import torch
import math
from torch.utils.data import Dataset
import pytorch_lightning as pl

class MarcoDataset(Dataset):
    """
    Dataset abstraction for MS MARCO document re-ranking. 
    """
    def __init__(self, data_dir, mode, tokenizer, max_seq_len=512, args=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        sample_size = 100
        # load queries
        self.queries = pd.read_csv(os.path.join(self.data_dir, f'msmarco-doc{mode}-queries.tsv'),
                                   sep='\t', header=None, names=['qid', 'query_text'], index_col='qid')
        if self.mode != 'test':
            self.relations = pd.read_csv(os.path.join(self.data_dir, f'msmarco-doc{mode}-qrels.tsv'),
                                   sep=' ', header=None, names=['qid', '0', 'did', 'label'])
        self.top100 = pd.read_csv(os.path.join(self.data_dir, f'msmarco-doc{mode}-top100'),
                                   sep=' ', header=None, names=['qid', 'Q0', 'did', 'rank', 'score', 'run'])
        self.doc_seek = pd.read_csv(os.path.join(self.data_dir,'msmarco-docs-lookup.tsv'),
                                        sep='\t', header=None,
                                        names=['did', 'trec_offset', 'tsv_offset'],
                                        index_col='did')
        self.docs = None
        
        # downsample the dataset so the positive:negative ratio is 1:10
        if mode == 'train':
           ''' qids = []
            with open(os.path.join(data_dir, 'msmarco-doctrain-selected-queries.txt'), "r") as file:
                lines = file.readlines()
                count = 0
                for line in lines:
                    count += 1
                    qids.append(int(line))
                    if count >= sample_size:
                        break
           '''#self.top100 = self.top100.loc[self.top100['qid'].isin(qids)]
           self.top100 = self.top100.sample(frac=0.1, random_state=42).append(self.relations[['qid', 'did']],ignore_index=True)
           self.top100.drop_duplicates(keep='first', inplace=True)
           # shuffle the data so positives are ~ evenly distributed
           self.top100 = self.top100.sample(frac=1, random_state=42).reset_index(drop=True) 
           # self.docs = pd.read_csv(os.path.join(data_dir, 'msmarco-docstrain.tsv'), 
           #                     sep='\t', header=None, names=['qid', 'did', 'qtext', 'dtext'], encoding='utf-8')
                                        
        elif mode == 'dev' and args.use_10_percent_of_dev:
            # use 10% of the data for dev during training
            import numpy as np; np.random.seed(42)
            queries = self.top100['qid'].unique()
            queries = np.random.choice(queries, int(len(queries)/50), replace=False)
            print(len(queries))
            self.top100 = self.top100[self.top100['qid'].isin(queries)]
            #self.docs = pd.read_csv(os.path.join(data_dir, 'msmarco-docsdev.tsv'), 
            #                    sep='\t', header=None, names=['qid', 'did', 'qtext', 'dtext'], encoding='utf-8')

        print(f'{mode} set len:', len(self.top100))

    # needed for map-style torch Datasets
    def __len__(self):
        return len(self.top100)

    # needed for map-style torch Datasets
    def __getitem__(self, idx):
        x = self.top100.iloc[idx]
        query = self.queries.loc[x.qid].query_text
        label = 0
        #if self.mode == 'train' or self.mode == 'dev':
        if self.mode=='n':
            docentry = self.docs.loc[(self.docs['qid'] == x.qid) & (self.docs['did'] == x.did)]
            try:
                document = 'N/A' if docentry.empty or ((type(docentry.dtext.item()) != str) and math.isnan(docentry.dtext)) else docentry.dtext.item()
            except:
                document = 'N/A'
            label = 0 if self.relations.loc[(self.relations['qid'] == x.qid) & (self.relations['did'] == x.did)].empty else 1
        else:
            doc_file = open(os.path.join(self.data_dir, 'msmarco-docs.tsv'), 'r', encoding='utf-8')
            # doc_file = open(os.path.join(self.data_dir, 'msmarco-docs.tsv'), 'r')
            file_offset = self.doc_seek.loc[x.did].tsv_offset
            doc_file.seek(file_offset, 0)
            line = doc_file.readline()
            doc_file.close()
            splited = line.split('\t')
            # when using num_workers > 1 seek get's fucked up
            assert(splited[0] == x.did)
            document = ' '.join(splited[3:]) 
            label = 0 if self.relations.loc[(self.relations['qid'] == x.qid) & (self.relations['did'] == x.did)].empty else 1
        tensors = self.one_example_to_tensors(query, document, idx, label)
        return tensors

    # main method for encoding the example
    def one_example_to_tensors(self, query, document, idx, label):

        encoded = self.tokenizer.encode_plus(query, document,
                        add_special_tokens=True,
                        max_length=self.max_seq_len,
                        truncation='only_second',
                        truncation_strategy='only_second',
                        return_overflowing_tokens=False,
                        return_special_tokens_mask=False,
                        return_token_type_ids=True,
                        pad_to_max_length=True
                            )
        encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])

        encoded['input_ids'] = torch.tensor(encoded['input_ids'])

        encoded.update({'label': torch.LongTensor([label]),
                        'idx': torch.tensor(idx)})
        return encoded

