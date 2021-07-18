import numpy as np


class DataLoader(object):
    def __init__(self, fpath1, fpath2, maxlen1, maxlen2, vocab_fpath):
        self.sents1, self.sents2 = self.load_data(
            fpath1, fpath2, maxlen1, maxlen2)
        self.token2idx, self.idx2token = self.load_vocab(vocab_fpath)
        self.maxlen1 = maxlen1
        self.maxlen2 = maxlen2

    def load_vocab(self, vocab_fpath):
        '''Loads vocabulary file and returns idx<->token maps
        vocab_fpath: string. vocabulary file path.
        Note that these are reserved
        0: <pad>, 1: <unk>, 2: <s>, 3: </s>

        Returns
        two dictionaries.
        '''
        vocab = [line.split()[0] for line in open(
            vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def load_data(self, fpath1, fpath2, maxlen1, maxlen2):
        '''Loads source and target data and filters out too lengthy samples.
        fpath1: source file path. string.
        fpath2: target file path. string.
        maxlen1: source sent maximum length. scalar.
        maxlen2: target sent maximum length. scalar.

        Returns
        sents1: list of source sents
        sents2: list of target sents
        '''
        sents1, sents2 = [], []
        with open(fpath1, 'r', encoding='utf-8') as f1, open(fpath2, 'r', encoding='utf-8') as f2:
            for sent1, sent2 in zip(f1, f2):
                if len(sent1.split()) + 1 > maxlen1:
                    continue  # 1: </s>
                if len(sent2.split()) + 1 > maxlen2:
                    continue  # 1: </s>
                sents1.append(sent1.strip())
                sents2.append(sent2.strip())
        return sents1, sents2

    def encode(self, inp, type, dict):
        '''Converts string to number. Used for `generator_fn`.
        inp: 1d byte array.
        type: "x" (source side) or "y" (target side)
        dict: token2idx dictionary

        Returns
        list of numbers
        '''
        inp_str = inp
        if type == "x":
            tokens = inp_str.split() + ["</s>"]
        else:
            tokens = ["<s>"] + inp_str.split() + ["</s>"]

        x = [dict.get(t, dict["<unk>"]) for t in tokens]
        return x

    def make_epoch_data(self, batch_size, shuffle=False):
        import copy
        new_sents1 = copy.deepcopy(self.sents1)
        new_sents2 = copy.deepcopy(self.sents2)
        if shuffle:
            import random
            random.shuffle(new_sents1)
            random.shuffle(new_sents2)
        xs = [self.encode(sent1, "x", self.token2idx) for sent1 in new_sents1]
        ys = [self.encode(sent2, "y", self.token2idx) for sent2 in new_sents2]
        batch_xs = []
        batch_ys = []
        for i in range(0, len(xs), batch_size):
            start = i
            end = start + batch_size
            batch_xs.append(xs[start:end])
            batch_ys.append(ys[start:end])
        if len(batch_xs[-1]) != batch_size:
            batch_xs = batch_xs[:-1]
            batch_ys = batch_ys[:-1]
        self.cur_xs = batch_xs
        self.cur_ys = batch_ys
        self.batch_num = len(batch_xs)
        self.idx = 0

    def get_batch(self, fill_maxlen=True):
        if self.idx >= self.batch_num:
            assert False
        cur_batch_x = self.cur_xs[self.idx]
        cur_batch_y = self.cur_ys[self.idx]
        self.idx += 1

        if fill_maxlen:
            cur_largest_len_x = self.maxlen1
            cur_largest_len_y = self.maxlen2
        else:
            cur_largest_len_x = max([len(x) for x in cur_batch_x])
            cur_largest_len_y = max([len(y) for y in cur_batch_y])

        cur_batch_x = np.array([self.align(x, cur_largest_len_x)
                                for x in cur_batch_x]).astype(np.float32)
        cur_batch_y = np.array([self.align(y, cur_largest_len_y)
                                for y in cur_batch_y]).astype(np.float32)
        return (cur_batch_x, cur_largest_len_x), (cur_batch_y, cur_largest_len_y)

    def align(self, arr, length):
        ori_len = len(arr)
        if length > ori_len:
            return arr + [0] * (length - ori_len)
        else:
            return arr[:length]

    def get_pad(self):
        return self.token2idx["<pad>"]
