import os
import numpy as np
import re

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

with open(os.path.join('open your FB15K dataset file , e.g. Desktop/FB15k', 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

with open(os.path.join('open your FB15K dataset file , e.g. Desktop/FB15k', 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

train_triples = read_triple(os.path.join('open your FB15K dataset file , e.g. Desktop/FB15k', 'open your relation patterns dataset file , e.g. antisymmetry_test.txt'), entity2id, relation2id)

file = open('create your new file, e.g. /Desktop/test','w')

for line in train_triples:
        s = str(line).replace('(', '').replace(')', '').replace(",", '') + "\n"
        file.write(s)

file.close()

