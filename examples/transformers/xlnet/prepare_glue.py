import os
import sys
import shutil
import argparse
import tempfile
import urllib
import io
import numpy as np
import logging
import pickle
import hetu as ht
from glue_processor.glue import PROCESSORS, convert_examples_to_features

if sys.version_info >= (3, 0):
    import urllib.request
import zipfile

URLLIB=urllib
if sys.version_info >= (3, 0):
    URLLIB=urllib.request

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {"CoLA":'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
             "SST":'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
             "QQP":'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
             "STS":'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
             "MNLI":'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
             "QNLI":'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
             "RTE":'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
             "WNLI":'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
             "MRPC":'https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC/dev_ids.tsv',
             "diagnostic":'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'}

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if task == "MNLI":
        print("\tNote (12/10/20): This script no longer downloads SNLI. You will need to manually download and format the data to use SNLI.")
    data_file = "%s.zip" % task
    URLLIB.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")

def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            URLLIB.urlretrieve(MRPC_TRAIN, mrpc_train_file)
            URLLIB.urlretrieve(MRPC_TEST, mrpc_test_file)
        except urllib.error.HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
            io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

    try:
        URLLIB.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))
    except KeyError or urllib.error.HTTPError:
        print("\tError downloading standard development IDs for MRPC. You will need to manually split your data.")
        return

    dev_ids = []
    with io.open(os.path.join(mrpc_dir, "dev_ids.tsv"), encoding='utf-8') as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
         io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, \
         io.open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    print("\tCompleted!")

def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    URLLIB.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return

def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def gen_tensor_dataset(features):
    all_input_ids = np.array(
        [f.input_ids for f in features],
        dtype=np.int64,
    )
    all_input_mask = np.array(
        [f.input_mask for f in features],
        dtype=np.int64,
    )
    all_segment_ids = np.array(
        [f.segment_ids for f in features],
        dtype=np.int64,
    )
    all_label_ids = np.array(
        [f.label_id for f in features],
        dtype=np.int64,
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
    print("Saving train features into cached file %s",
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
    print("Saving dev features into cached file %s",
                cached_train_features_file)
    with open(cached_train_features_file, "wb") as writer:
        pickle.dump(feature_dic, writer)
    

def get_tokenizer(model="xlnet-base-cased", add_tokens=None):
    if not os.path.exists('spiece.model'):
        origin = ht.tokenizers.XLNetTokenizer.pretrained_vocab_files_map["vocab_file"][model]
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "spiece.model")

    tokenizer = ht.tokenizers.XLNetTokenizer('spiece.model')
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)
    return tokenizer
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='directory to save data to', type=str, default='./data/glue_data')
    parser.add_argument('-t', '--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='SST')
    parser.add_argument('--path_to_mrpc', help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    parser.add_argument('-m', '--max_seq_length', default=128, type=int, help="The maximum total input sequence length after WordPiece ")
      
    
    args = parser.parse_args()
    max_seq_length = args.max_seq_length
    
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)

        task = task.lower()
        if task =='sst':
            task = 'sst-2'
        processor = PROCESSORS[task]()
        
        add_tokens = None
        if task in {'mrpc', 'mnli'}:
            dir_name = task.upper()
            add_tokens = '<$>'
        elif task == 'cola':
            dir_name = 'CoLA'
        else:  # SST-2
            assert task == 'sst-2'
            dir_name = 'SST-2'
        data_dir = './data/glue_data/%s'%dir_name

        tokenizer = get_tokenizer(add_tokens=add_tokens)

        get_train_features(data_dir, task, max_seq_length, tokenizer, processor)
        get_dev_features(data_dir, task, max_seq_length, tokenizer, processor)
    