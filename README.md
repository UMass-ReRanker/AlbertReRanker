# AlbertReRanker for MS MARCO document re-ranking task

## About
We employ ALBERT, a lite variant of BERT consisting only on 1/15th number of paramaters as compared to thhat of BERT-base, on the MS MARCO document re-ranking dataset. 

## Training
The training process must be done in two parts - first sentence selection, second ALBERT fine tuning.
For sentence selection - 

1. Expected Directory Strucutre 

```
src 
└── sentence_selection.py
    msmarco-doctrain-selected-queries.txt
    msmarco-doctrain-top100.csv
    glove 
    ├── glove.840B.300d.txt    
    corpus
    ├── msmarco-docs.tsv
    collection_queries
    ├── queries.train.tsv   

```
2. Select Sentences - 
```
python sentence_selection.py
```

3. ALBERT Fine Tuning - 
```
python train_albert_on_msmarco.py --data_loader_bs=12 --trainer_batch_size=10 --data_dir='<MS-MARCO DATA DIR>' --max_seq_len=512 --val_check_interval=600 --lr=9e-6

```

## Credits
Code in this repository has been adapted from https://github.com/isekulic/longformer-marco. 
