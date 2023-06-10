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

with open(os.path.join('open your WN18RR dataset file , e.g. Desktop/WN18RR', 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

with open(os.path.join('open your WN18RR dataset file , e.g. Desktop/WN18RR', 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

train_triples = read_triple(os.path.join('open your WN18RR dataset file , e.g. Desktop/WN18RR', 'train.txt'), entity2id, relation2id)


file = open('create your new file, e.g. /Desktop/test','w')


for line in train_triples:
    if line[1] == 10:
        s = str(line).replace('(', '').replace(')', '').replace(",", '') + "\n"
        file.write(s)
    else:
        pass

file.close()


"""
WN18RR:

0	_hypernym
1	_derivationally_related_form
2	_instance_hypernym
3	_also_see
4	_member_meronym
5	_synset_domain_topic_of
6	_has_part
7	_member_of_domain_usage
8	_member_of_domain_region
9	_verb_group
10	_similar_to

"""

