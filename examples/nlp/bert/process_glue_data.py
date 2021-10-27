from glue.glue import PROCESSORS, convert_examples_to_features
import hetu
import numpy as np
import os
import logging
import pickle

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
    cache_path = './preprocessed_data/glue'
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
    cache_path = './preprocessed_data/glue'
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

task_name = 'sst-2'
processor = PROCESSORS[task_name]()

vocab_path = "./datasets/bert-base-uncased-vocab.txt"
do_lower_case = True
max_seq_length = 512

if task_name in {'mrpc', 'mnli'}:
    dir_name = task_name.upper()
elif task_name == 'cola':
    dir_name = 'CoLA'
else:  # SST-2
    assert task_name == 'sst-2'
    dir_name = 'SST-2'
data_dir = './datasets/glue/%s'%dir_name

tokenizer = hetu.BertTokenizer(
    vocab_file=vocab_path, 
    do_lower_case = do_lower_case,
    max_len=512,
)

get_train_features(data_dir, task_name, max_seq_length, tokenizer, processor)
get_dev_features(data_dir, task_name, max_seq_length, tokenizer, processor)

