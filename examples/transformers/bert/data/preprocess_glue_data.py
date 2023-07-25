from glue_processor.glue import PROCESSORS, convert_examples_to_features
import numpy as np
import os
import logging
import pickle
import argparse
from tokenization import BertTokenizer

def gen_tensor_dataset(features):
    all_input_ids = np.array(
        [f.input_ids for f in features],
        dtype=np.long,
    )
    all_input_mask = np.array(
        [f.input_mask for f in features],
        dtype=np.long,
    )
    all_segment_ids = np.array(
        [f.segment_ids for f in features],
        dtype=np.long,
    )
    all_label_ids = np.array(
        [f.label_id for f in features],
        dtype=np.long,
    )
    return {'input_ids': all_input_ids,
            'token_type_ids': all_segment_ids,
            'attention_mask': all_input_mask,
            'label_ids': all_label_ids}

def get_train_features(data_dir, task_name, max_seq_length, tokenizer, processor):
    cache_path = './data/preprocessed_glue_data/'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    cached_train_features_file = os.path.join(
        cache_path,
        '%s_train_features'%task_name,
    )
    train_features = None
    train_examples = processor.get_train_examples(data_dir)
    train_features, _ = convert_examples_to_features(
        train_examples,
        processor.get_labels(),
        max_seq_length,
        tokenizer,
    )
    feature_dic = gen_tensor_dataset(train_features)
    print("  Saving train features into cached file %s",
                cached_train_features_file)
    with open(cached_train_features_file, "wb") as writer:
        pickle.dump(feature_dic, writer)

def get_dev_features(data_dir, task_name, max_seq_length, tokenizer, processor):
    cache_path = './data/preprocessed_glue_data/'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    cached_train_features_file = os.path.join(
        cache_path,
        '%s_dev_features'%task_name,
    )
    dev_features = None
    dev_examples = processor.get_dev_examples(data_dir)
    dev_features, _ = convert_examples_to_features(
        dev_examples,
        processor.get_labels(),
        max_seq_length,
        tokenizer,
    )
    feature_dic = gen_tensor_dataset(dev_features)
    print("  Saving dev features into cached file %s",
                cached_train_features_file)
    with open(cached_train_features_file, "wb") as writer:
        pickle.dump(feature_dic, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='sst-2',
        type=str,
        required=True,
        choices=PROCESSORS.keys(),
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece "
        "tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    processor = PROCESSORS[task_name]()

    vocab_path = "./data/bert-base-uncased-vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length

    if task_name in {'mrpc', 'mnli'}:
        dir_name = task_name.upper()
    elif task_name == 'cola':
        dir_name = 'CoLA'
    else:  # SST-2
        assert task_name == 'sst-2'
        dir_name = 'SST-2'
    data_dir = './data/download/glue/%s'%dir_name

    tokenizer = BertTokenizer(
        vocab_file=vocab_path, 
        do_lower_case = do_lower_case,
        max_len=512,
    )

    get_train_features(data_dir, task_name, max_seq_length, tokenizer, processor)
    get_dev_features(data_dir, task_name, max_seq_length, tokenizer, processor)