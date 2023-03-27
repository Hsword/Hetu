import os
import requests
import numpy as np
import hetu as ht
from six.moves import urllib
import argparse
from tqdm import tqdm
import random
import collections
import h5py

def random_mask(ids, rate, mask_id):
    mask_ids, labels = [], []
    for token_id in ids:
        mmm = random.random()
        if mmm <= rate:
            mask_ids.append(mask_id)
            labels.append(token_id)
        else:
            mask_ids.append(token_id)
            labels.append(-100)  
    return mask_ids, labels


def get_tokenizer(model='allenai/longformer-base-4096'):
    if not os.path.exists('vocab.json'):
        origin = ht.tokenizers.LongformerTokenizer.pretrained_vocab_files_map["vocab_file"][model]
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "vocab.json")
    if not os.path.exists('merges.txt'):
        origin = ht.tokenizers.LongformerTokenizer.pretrained_vocab_files_map["merges_file"][model]
        print('Downloading merges from %s' % origin)
        urllib.request.urlretrieve(origin, "merges.txt")        
    tokenizer = ht.tokenizers.LongformerTokenizer('vocab.json', 'merges.txt')
    return tokenizer

            
def prepare(args):
    dataset = args.dataset
    input_path = './data/%s/'%dataset
    seq_len = 512
    tokenizer = get_tokenizer()
    mask_rate = 0.15
    mask_token_id = tokenizer.mask_token_id
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if dataset=='shakespeare':
        input_file_path = input_path + 'input.txt'
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)
        
        with open(input_file_path, 'r') as f:
            texts = f.readlines()
        n = len(texts)
        
        train_data = collections.OrderedDict()
        train_data["input_ids"] = []
        train_data["attention_mask"] = []
        train_data["labels"] = []

        for i in range(n):
            text = texts[i]
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)     
            mask_ids, labels = random_mask(ids, mask_rate, mask_token_id) 
            if(np.sum(labels) == len(labels) *(-100)):
                continue
            if len(mask_ids) < seq_len:
                att_mask = [1] * len(mask_ids) + [0] * (seq_len - len(mask_ids))
            else:
                att_mask = [1] * seq_len
            mask_ids = mask_ids[:seq_len]
            labels = labels[:seq_len]

            masked_data = mask_ids + [tokenizer.pad_token_id] * (seq_len - len(mask_ids))
            labels = labels + [-100] * (seq_len - len(labels))
            
            train_data["input_ids"].append(masked_data)
            train_data["attention_mask"].append(att_mask)
            train_data["labels"].append(labels)
            
        print("saving data......")
        output_file = input_path + 'train.h5py'
        f= h5py.File(output_file, 'w')
        f.create_dataset("input_ids", data=np.array(train_data["input_ids"], dtype=np.int32), dtype='i4', compression='gzip')
        f.create_dataset("attention_mask", data=np.array(train_data["attention_mask"], dtype=np.int32), dtype='i1', compression='gzip')
        f.create_dataset("labels", data=np.array(train_data["labels"], dtype=np.int32), dtype='i4', compression='gzip')
        f.flush()
        f.close()
        
    elif dataset=='openwebtext':
        
        from datasets import load_dataset
        num_proc = args.num_proc
        dataset = load_dataset("openwebtext")
        split_dataset = dataset["train"]
        
        def process(data):
            if data['text']=='':
                return  {'ids': [tokenizer.eos_token_id], 'len': 1}
                
            tokens = tokenizer.tokenize(data['text'])
            ids = tokenizer.convert_tokens_to_ids(tokens)     
            ids.append(tokenizer.eos_token_id)
            out = {'ids': ids, 'len': len(ids)}
            return out
        
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        
        train_data = collections.OrderedDict()
        train_data["input_ids"] = []
        train_data["attention_mask"] = []
        train_data["labels"] = []

        for example in tqdm(tokenized):
            ids = example['ids']
            mask_ids, labels = random_mask(ids, mask_rate, mask_token_id) 
            if len(mask_ids) < seq_len:
                att_mask = [1] * len(mask_ids) + [0] * (seq_len - len(mask_ids))
            else:
                att_mask = [1] * seq_len
            mask_ids = mask_ids[:seq_len]
            labels = labels[:seq_len]
    
            masked_data = mask_ids + [tokenizer.pad_token_id] * (seq_len - len(mask_ids))
            labels = labels + [-100] * (seq_len - len(labels))
                
            train_data["input_ids"].append(masked_data)
            train_data["attention_mask"].append(att_mask)
            train_data["labels"].append(labels)
    
        print("saving data......")
        output_file = input_path + 'train.h5py'
        f= h5py.File(output_file, 'w')
        f.create_dataset("input_ids", data=np.array(train_data["input_ids"], dtype=np.int32), dtype='i4', compression='gzip')
        f.create_dataset("attention_mask", data=np.array(train_data["attention_mask"], dtype=np.int32), dtype='i1', compression='gzip')
        f.create_dataset("labels", data=np.array(train_data["labels"], dtype=np.int32), dtype='i4', compression='gzip')
        f.flush()
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset', type=str, default='shakespeare', help='Dataset used to train.'
    )
    parser.add_argument(
        '--num_proc', type=int, default=8, help='Number of processes.'
    )
    args = parser.parse_args()

    prepare(args)



