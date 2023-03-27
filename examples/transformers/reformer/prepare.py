import os
import requests
import numpy as np
import hetu as ht
from six.moves import urllib
import argparse
from tqdm import tqdm

def tokenize(text, model='facebook/bart-base', add_eos=False):
    if not os.path.exists('vocab.json'):
        origin = ht.tokenizers.BartTokenizer.pretrained_vocab_files_map["vocab_file"][model]
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "vocab.json")
    if not os.path.exists('merges.txt'):
        origin = ht.tokenizers.BartTokenizer.pretrained_vocab_files_map["merges_file"][model]
        print('Downloading merges from %s' % origin)
        urllib.request.urlretrieve(origin, "merges.txt")        
    tokenizer = ht.tokenizers.BartTokenizer('vocab.json', 'merges.txt')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)       
    if add_eos:
        ids.append(tokenizer.eos_token_id)
    return ids
            
def prepare(args):
    dataset = args.dataset
    input_path = './data/%s/'%dataset
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if dataset=='shakespeare':
        input_file_path = input_path + 'input.txt'
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)
        
        with open(input_file_path, 'r') as f:
            data = f.read()
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]
        
        train_ids = tokenize(train_data)
        val_ids = tokenize(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")
        
        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile('./data/shakespeare/train.bin')
        val_ids.tofile('./data/shakespeare/test.bin')
    elif dataset=='openwebtext':
        
        from datasets import load_dataset
        num_proc = args.num_proc
        dataset = load_dataset("openwebtext")
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=123, shuffle=True)

        def process(data):
            ids = tokenize(data['text'], add_eos=True)
            out = {'ids': ids, 'len': len(ids)}
            return out
        
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        

        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = f'./data/openwebtext/{split}.bin'
            dtype = np.uint16 
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()

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



