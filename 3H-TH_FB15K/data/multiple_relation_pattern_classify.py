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

with open(os.path.join('open your FB15K dataset file , e.g. /Desktop/FB15k', 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

train_triples = read_triple(os.path.join('open your FB15K dataset file , e.g. Desktop/FB15k', 'test.txt'), entity2id, relation2id)


file = open('create your new file, e.g. /Desktop/test','w')


save_triples = []
save_line = []
for line in train_triples:
    if line[0] not in save_triples:
        head_multi = line[0]
        save_triples.append(head_multi)
        save_1 = []
        save_1.append(line)
        for lines in train_triples:
            if lines[0] == head_multi and lines != line:
                save_1.append(lines)
            else:
                pass
        for liness in save_1:
            save_2 = []
            tail_multi = liness[2]
            for linesss in save_1:
                if linesss[2] == tail_multi and linesss != liness and linesss not in save_line:
                    save_line.append(linesss)
                    save_2.append(linesss)
                    s_2 = str(linesss).replace('(', '').replace(')', '').replace(",", '') + "\n"
                    file.write(s_2)
                else:
                    pass
            if save_2 != []:
                s_1 = str(liness).replace('(', '').replace(')', '').replace(",", '') + "\n"
                file.write(s_1)
                save_line.append(liness)
            else:
                pass
    else:
        pass


file.close








"""
for line in train_triples:
    if line[0] == 7315:
        s = str(line).replace('(', '').replace(')', '').replace(",", '') + "\n"
        file.write(s)
    else:
        pass

file.close()
"""