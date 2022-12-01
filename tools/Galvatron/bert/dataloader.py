import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class DataLoaderForBert(Dataset):
    def __init__(self):
        f = h5py.File('./data/sample_wiki_data_for_bert.hdf5', "r")
        hdf5_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels', 'next_sentence_label']
        self.input_ids, self.attention_mask, self.token_type_ids, self.masked_lm_labels, \
                self.next_sentence_label = [np.tile(a, [16]+[1]*len(a.shape[1:])) for a in [np.asarray(f[key][:]) for key in hdf5_keys]]
        self.next_sentence_label = self.next_sentence_label.reshape(-1, 1)
        self.dataset_size = self.input_ids.shape[0]
        f.close()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_mask = torch.LongTensor(self.attention_mask[idx])
        token_type_ids = torch.LongTensor(self.token_type_ids[idx])
        masked_lm_labels = torch.LongTensor(self.masked_lm_labels[idx])
        next_sentence_label = torch.LongTensor(self.next_sentence_label[idx])
        return input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label

class DataLoaderForBert_wrapped(Dataset):
    def __init__(self):
        f = h5py.File('./data/sample_wiki_data_for_bert.hdf5', "r")
        hdf5_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels', 'next_sentence_label']
        self.input_ids, self.attention_mask, self.token_type_ids, self.masked_lm_labels, \
                self.next_sentence_label = [np.tile(a, [16]+[1]*len(a.shape[1:])) for a in [np.asarray(f[key][:]) for key in hdf5_keys]]
        self.next_sentence_label = self.next_sentence_label.reshape(-1, 1)
        self.dataset_size = self.input_ids.shape[0]
        f.close()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_mask = torch.LongTensor(self.attention_mask[idx])
        token_type_ids = torch.LongTensor(self.token_type_ids[idx])
        masked_lm_labels = torch.LongTensor(self.masked_lm_labels[idx])
        next_sentence_label = torch.LongTensor(self.next_sentence_label[idx])
        return (input_ids, attention_mask, token_type_ids), (masked_lm_labels, next_sentence_label)