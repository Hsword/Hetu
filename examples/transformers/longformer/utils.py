import numpy as np
import os
import pickle
import h5py
import random

class DataLoader(object):
    def __init__(self, input_file, batch_size):
        self.input_file = input_file
        self.batch_size = batch_size
        self.data_names = ['input_ids', 'attention_mask', 'labels']
        self.data = {'input_ids':[],
                    'attention_mask':[],
                    'labels':[]}
        self.batch_data = {'input_ids':[],
                    'attention_mask':[],
                    'labels':[]}
        self.cur_batch_data = {'input_ids':[],
                    'attention_mask':[],
                    'labels':[]}
        f = h5py.File(input_file, "r")
        hdf5_keys = ['input_ids', 'attention_mask', 'labels']
        self.data['input_ids'], self.data['attention_mask'], self.data['labels'] = [np.asarray(f[key][:]) for key in hdf5_keys]
        f.close()
        self.data_len = self.data['input_ids'].shape[0]

        print('Successfully loaded data file %s!'%input_file)
        self.make_epoch_data()

    def make_epoch_data(self):
        for i in range(0, self.data_len, self.batch_size):
            start = i
            end = start + self.batch_size
            if end > self.data_len:
                end = self.data_len
            if end-start != self.batch_size:
                break
            for data_name in self.data_names:
                self.batch_data[data_name].append(self.data[data_name][start:end]) 

        self.batch_num = len(self.batch_data['input_ids'])
    
    def get_batch(self, idx):
        if idx >= self.batch_num:
            assert False
        for data_name in self.data_names:
            self.cur_batch_data[data_name] = self.batch_data[data_name][idx]

        return self.cur_batch_data.copy()

        
class DataLoaderForGlue(object):
    def __init__(self, task_name='sst-2', batch_size = 1024, datatype='train'):
        self.data_names = ['input_ids','token_type_ids','attention_mask','label_ids']
        self.data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.batch_size=batch_size
        self.batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.cur_batch_data = {'input_ids':[],
                    'token_type_ids':[],
                    'attention_mask':[],
                    'label_ids':[]}
        self.load_data(task_name=task_name, datatype=datatype)
        self.make_epoch_data()

    def load_data(self, task_name='sst-2', datatype='train'):
        print('Loading preprocessed dataset %s...'%task_name)
        cached_train_features_file = os.path.join('./data/preprocessed_glue_data/','%s_%s_features'%(task_name,datatype))

        try:
            with open(cached_train_features_file, "rb") as reader:
                self.data = pickle.load(reader)
            print("Loaded pre-processed features from {}".format(
                cached_train_features_file))
        except:
            print("Did not find pre-processed features from {}".format(
                cached_train_features_file))
            print("Please run sh scripts/create_datasets_from_start.sh first!")
            assert False

        self.data_len = self.data['input_ids'].shape[0]
        self.num_labels = np.max(self.data['label_ids'])+1
        print(self.data['input_ids'].shape)
        print('Successfully loaded GLUE dataset %s for %s!'%(task_name,datatype))
    
    def make_epoch_data(self):
        for i in range(0, self.data_len, self.batch_size):
            start = i
            end = start + self.batch_size
            if end > self.data_len:
                end = self.data_len
            if end-start != self.batch_size:
                break
            for data_name in self.data_names:
                self.batch_data[data_name].append(self.data[data_name][start:end]) 

        self.batch_num = len(self.batch_data['input_ids'])
    
    def get_batch(self, idx):
        if idx >= self.batch_num:
            assert False
        for data_name in self.data_names:
            self.cur_batch_data[data_name] = self.batch_data[data_name][idx]

        return self.cur_batch_data.copy() 