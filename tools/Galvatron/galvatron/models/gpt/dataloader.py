import torch
from torch.utils.data import Dataset
import numpy as np

class DataLoaderForGPT(Dataset):
    def __init__(self, args, device):
        self.vocab_size = args.vocab_size
        self.sentence_length = args.seq_length
        self.dataset_size = 2560 * 16
        self.data_length = np.random.randint(1,self.sentence_length+1,(self.dataset_size,))
        self.device = device

        self.input_ids = []
        for i in range(self.dataset_size):
            sentence = np.random.randint(0,self.vocab_size,(self.sentence_length,))
            sentence[self.data_length[i]:] = 0
            mask = np.ones((self.sentence_length,))
            mask[self.data_length[i]:] = 0
            self.input_ids.append(sentence)
        
        self.input_ids = np.array(self.input_ids)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx]).to(self.device)
        return input_ids