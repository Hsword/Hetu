import json
import ast
import numpy as np
import nltk
# all products with metadata
filemap = {'train': './amazon-3M_train_map.txt',
           'test': './amazon-3M_test_map.txt',
           'bow': './amazon-3M_feature_map.txt',
           'meta': './metadata.json',
           'output': './graph.npz',
           'output_sparse': './sparsefeature.npy'}


def getBagofWord():
    bow = dict()
    with open(filemap['bow'], 'r') as f:
        # start with 1, 0 for padding
        word_cnt = 1
        for line in f.read().strip().split():
            bow[line] = word_cnt
            word_cnt += 1
    return bow


def gettoken(descriptions, length):
    bow = getBagofWord()
    token_matrix = []
    for desc in descriptions:
        token_id = []
        token = nltk.word_tokenize(desc.lower())
        for word in token:
            if word in bow:
                token_id.append(bow[word])
                if len(token_id) == length:
                    break
        while len(token_id) < length:
            token_id.append(0)
        token_matrix.append(token_id)
    return np.array(token_matrix)


prod_all = dict()
prod_rcd = dict()
with open(filemap['meta'], 'r') as f:
    for line in f:
        prod = ast.literal_eval(line.strip().replace('\n', '\\n'))
        asin = prod['asin']
        prod_all[asin] = prod
        if 'related' in prod and 'categories' in prod and 'description' in prod:
            prod_rcd[asin] = prod

testNodes = set()
prod_gcn = dict()
asin2id = dict()
cnt_id = 0
asinlist = []

for kword in ['train', 'test']:
    with open(filemap[kword], 'r') as f:
        for line in f:
            asin = line.split()[0]
            if asin in prod_rcd:
                if kword == 'test':
                    testNodes.add(asin)
                prod_gcn[asin] = prod_rcd[asin]
                asin2id[asin] = cnt_id
                cnt_id += 1
                asinlist.append(asin)

graphlen = len(prod_gcn)
print('#products with rel/cat/des/feat (GCN assumptions)', graphlen)
print('#trainNodes:', graphlen-len(testNodes), 'testNodes:', len(testNodes))

print(len(asin2id))

cat2id = dict()
cnt_id = 0

class_map = np.zeros(graphlen).astype(np.int32)
train_map = np.zeros(graphlen).astype(np.int32)
descriptions = []
for idx, asin in enumerate(asinlist):
    prod = prod_gcn[asin]
    isTest = True if asin in testNodes else False

    cat = prod['categories'][0][0]
    if cat not in cat2id:
        cat2id[cat] = (cnt_id, 0, 0)
        cnt_id += 1

    if isTest:
        cat2id[cat] = (cat2id[cat][0], cat2id[cat][1], cat2id[cat][2]+1)
    else:
        cat2id[cat] = (cat2id[cat][0], cat2id[cat][1]+1, cat2id[cat][2])

    class_map[idx] = cat2id[cat][0]
    train_map[idx] = 0 if isTest else 1
    if "title" in prod:
        descriptions.append(prod["title"] + " " + prod['description'])
    else:
        descriptions.append(prod['description'])

print('Classes:', cat2id)
print("Num Classes:", len(cat2id))

links_set = set()
for idx, asin in enumerate(asinlist):
    for rel, neighbors in prod_gcn[asin]['related'].items():
        for asin_nei in neighbors:
            if asin_nei not in asin2id:
                continue
            idx_nei = asin2id[asin_nei]
            lk = (idx, idx_nei) if idx_nei > idx else (idx_nei, idx)
            if lk not in links_set:
                links_set.add(lk)
links = np.array(list(links_set))
print('#links between products:', len(links))
token_matrix = gettoken(descriptions, 16)
np.savez(file=filemap['output'], y=class_map, train_map=train_map, edge=links)
np.save(file=filemap['output_sparse'], arr=token_matrix)
