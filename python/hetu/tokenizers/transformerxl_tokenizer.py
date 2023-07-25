# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Tokenization classes for Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl.
"""

import logging
import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
from .utils import PreTrainedTokenizer
from tokenizers import AddedToken
import numpy as np
import importlib.util

if importlib.util.find_spec("sacremoses") is not None:
    import sacremoses as sm
else:
    assert False, "Pip install sacremoses is all you need"


VOCAB_FILES_NAMES = {
    "pretrained_vocab_file": "vocab.pkl",
    "pretrained_vocab_file_torch": "vocab.bin",
    "vocab_file": "vocab.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "pretrained_vocab_file": {
        "transfo-xl-wt103": "https://huggingface.co/transfo-xl-wt103/resolve/main/vocab.pkl",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "transfo-xl-wt103": None,
}

PRETRAINED_CORPUS_ARCHIVE_MAP = {
    "transfo-xl-wt103": "https://huggingface.co/transfo-xl-wt103/resolve/main/corpus.bin",
}
CORPUS_NAME = "corpus.bin"

MATCH_NUMBERS = r"(?<=\d)[,.](?=\d)", r" @\g<0>@ "
DETOKENIZE_NUMBERS = [(r" @\,@ ", r","), (r" @\.@ ", r".")]


def tokenize_numbers(text_array: List[str]) -> List[str]:
    """
    Splits large comma-separated numbers and floating point values. This is done by replacing commas with ' @,@ ' and
    dots with ' @.@ '.

    Args:
        text_array: An already tokenized text as list.

    Returns:
        A list of strings with tokenized numbers.

    Example:

    ```python
    >>> tokenize_numbers(["$", "5,000", "1.73", "m"])
    ["$", "5", "@,@", "000", "1", "@.@", "73", "m"]
    ```"""
    tokenized = []
    for i in range(len(text_array)):
        reg, sub = MATCH_NUMBERS
        replaced = re.sub(reg, sub, text_array[i]).split()
        tokenized.extend(replaced)

    return tokenized


def detokenize_numbers(text: str) -> str:
    """
    Inverts the operation of *tokenize_numbers*. This is replacing ' @,@ ' and ' @.@' by ',' and '.'.

    Args:
        text: A string where the number should be detokenized.

    Returns:
        A detokenized string.

    Example:

    ```python
    >>> detokenize_numbers("$ 5 @,@ 000 1 @.@ 73 m")
    "$ 5,000 1.73 m"
    ```"""
    for reg, sub in DETOKENIZE_NUMBERS:
        text = re.sub(reg, sub, text)
    return text


class TransfoXLTokenizer(PreTrainedTokenizer):
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        special=None,
        min_freq=0,
        max_size=None,
        lower_case=False,
        delimiter=None,
        vocab_file=None,
        pretrained_vocab_file: str = None,
        never_split=None,
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=["<formula>"],
        language="en",
        **kwargs
    ):
        self.special=special,
        self.min_freq=min_freq,
        self.max_size=max_size,
        self.lower_case=lower_case,
        self.delimiter=delimiter,
        self.vocab_file=vocab_file,
        self.pretrained_vocab_file=pretrained_vocab_file,
        self.never_split=never_split,
        self.unk_token=unk_token,
        self.eos_token=eos_token,
        self.additional_special_tokens=additional_special_tokens,
        self.language=language,


        if special is None:
            special = []
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        
        self.punctuation_symbols = '!"#$%&()*+,-./\\:;<=>?@[\\]^_`{|}~'
        self.punction_without_space_before_pattern = re.compile(rf"[^\s][{self.punctuation_symbols}]")
        self.punctuation_with_space_around_pattern = self._compile_space_around_punctuation_pattern()
        self.language = language
        self.moses_punct_normalizer = sm.MosesPunctNormalizer(language)
        self.moses_tokenizer = sm.MosesTokenizer(language)
        self.moses_detokenizer = sm.MosesDetokenizer(language)
        self.added_tokens_encoder = {}

        # This try... catch... is not beautiful but honestly this tokenizer was not made to be used
        # in a library like ours, at all.
        try:
            vocab_dict = None
            if pretrained_vocab_file is not None:
                with open(pretrained_vocab_file, "rb") as f:
                    vocab_dict = pickle.load(f)

            if vocab_dict is not None:
                for key, value in vocab_dict.items():
                    if key not in self.__dict__:
                        self.__dict__[key] = value
            elif vocab_file is not None:
                self.build_vocab()

        except Exception as e:
            raise ValueError(
                f"Unable to parse file {pretrained_vocab_file}. Unknown format. "
                "If you tried to load a model saved through TransfoXLTokenizerFast, "
                "please note they are not compatible."
            ) from e

        if vocab_file is not None:
            self.build_vocab()
                
        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            **kwargs,
        )
        if never_split is None:
            never_split = self.all_special_tokens
        self.never_split = never_split

    @property
    def do_lower_case(self):
        return self.lower_case

    def _compile_space_around_punctuation_pattern(self):
        look_ahead_for_special_token = f"(?=[{self.punctuation_symbols}])"
        look_ahead_to_match_all_except_space = r"(?=[^\s])"
        return re.compile(r"" + look_ahead_for_special_token + look_ahead_to_match_all_except_space)

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose:
            logger.info(f"counting file {path} ...")
        assert os.path.exists(path), f"Input file {path} not found"

        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            logger.info(f"counting {len(sents)} sents ...")
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        if "<UNK>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<UNK>"]
        elif "<unk>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<unk>"]
        else:
            raise ValueError("No <unknown> token in vocabulary")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["pretrained_vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "wb") as f:
            pickle.dump(self.__dict__, f)
        return (vocab_file,)

    def build_vocab(self):
        if self.vocab_file:
            logger.info(f"building vocab from {self.vocab_file}")
            self._build_from_file(self.vocab_file)
            logging.info(f"final vocab size {len(self)}")
        else:
            logging.info(f"building vocab with min_freq={self.min_freq}, max_size={self.max_size}")
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            logging.info(f"final vocab size {len(self)} from {len(self.counter)} unique tokens")

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, f"{sym.strip('<>')}_idx", self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def moses_punct_norm(self, text):
        return self.moses_punct_normalizer.normalize(text)

    def moses_tokenize(self, text):
        return self.moses_tokenizer.tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split
        )

    def moses_pipeline(self, text: str) -> List[str]:
        text = self.moses_punct_norm(text)
        text = self.moses_tokenize(text)
        text = tokenize_numbers(text)
        return text

    def _convert_id_to_token(self, idx):
        """Converts an id in a token (BPE) using the vocab."""
        assert 0 <= idx < len(self), f"Index {idx} out of vocabulary range"
        return self.idx2sym[idx]

    def _convert_token_to_id(self, sym):
        """Converts a token (str) in an id using the vocab."""
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # logging.info(f'encounter unk {sym}')
            # assert '<eos>' not in sym
            if hasattr(self, "unk_idx"):
                return self.sym2idx.get(sym, self.unk_idx)
            # Backward compatibility with pre-trained models
            elif "<unk>" in self.sym2idx:
                return self.sym2idx["<unk>"]
            elif "<UNK>" in self.sym2idx:
                return self.sym2idx["<UNK>"]
            else:
                raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement")

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string. Additionally, the split numbers are converted back
        into it's original form.
        """
        out_string = self.moses_detokenizer.detokenize(tokens)
        return detokenize_numbers(out_string).strip()

    @property
    def vocab_size(self):
        return len(self.idx2sym)

    def get_vocab(self):
        return dict(self.sym2idx, **self.added_tokens_encoder)

    def _tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == "":
            symbols = line
        else:
            symbols = self.moses_pipeline(line)

        if add_double_eos:  # lm1b
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols
