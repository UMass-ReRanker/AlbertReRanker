import numpy as np
from scipy import spatial
from datetime import datetime
import pandas as pd
import os
import copy

max_doc_len = 500

glove_embeddings = {}
embeds_file = open('glove/glove.840B.300d.txt', 'r')
for line in embeds_file:
    try:
        splitLines = line.split()
        word = splitLines[0]
        vector = np.array([float(value) for value in splitLines[1:]])
        glove_embeddings[word] = vector
    except:
        continue

embeds_file.close()

def get_embedding(text):
    words = text.split(' ')
    count_words = 0
    text_vector = np.zeros((300), dtype='float32')
    for word in words:
        if word in glove_embeddings:
            text_vector += glove_embeddings[word]
            count_words += 1
    if count_words > 0:
        text_vector /= count_words
    return text_vector

top100 = pd.read_csv('msmarco-doctrain-top100', sep=' ', header=None, names=['qid', 'q0', 'did', 'rank', 'score', 'model'])
corpus = pd.read_csv(os.path.join('corpus', f'msmarco-docs.tsv'), sep='\t', header=None, names=['did', 'url', 'title', 'body'])

all_queries = open(os.path.join('collection_queries', 'queries.train.tsv'), 'r')
queries_dict = {}
queries_content = all_queries.readlines()
for line in queries_content:
    qid, query = line.split('\t')
    queries_dict[qid] = query

output_file = open('dp_0.output', 'w')

queries = []
queries_file = open('msmarco-doctrain-selected-queries.txt', 'r')
q_line = queries_file.readlines()[:2500]
for qid in q_line:
    qid = qid.rstrip()
    docs_list = top100.loc[top100['qid'] == int(qid)]
    query_vector = get_embedding(queries_dict[qid])
    for index, row in docs_list.iterrows():
        doc_id = row['did']
        data = corpus.loc[corpus['did'] == doc_id]
        doc_content = data['body'].values[0]
        try:
            sentences = doc_content.split('.')
        except:
            continue
        sentences_scores = [[0]*3 for _ in range(len(sentences))]

        for (i, sentence) in enumerate(sentences):
            sentences_scores[i][0] = i
            sentences_scores[i][1] = len(sentence.split(' '))
            sentences_scores[i][2] = 1 - spatial.distance.cosine(query_vector, get_embedding(sentence))
        sentences_scores = sorted(sentences_scores, key=lambda x: -x[2])
        final_doc = ""
        new_doc_len = 0
        idx = 0
        while idx < len(sentences) and (new_doc_len + sentences_scores[idx][1]) < 512:
            final_doc += (sentences[sentences_scores[idx][0]] + '. ')
            new_doc_len += (sentences_scores[idx][1])
            idx += 1
        output_file.write(str(qid) + '\t' + str(doc_id) + '\t' + queries_dict[qid].rstrip() + '\t' + final_doc + '\n')
        print(doc_id)
output_file.close()            
