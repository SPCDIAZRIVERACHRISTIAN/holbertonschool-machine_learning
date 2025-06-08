#!/usr/bin/env python3
"""
Load, tokenize tensorflow Dataset
"""

import tensorflow_datasets as tfds

class Dataset:
    """
    Loads and prepares the TED HRLR translation dataset for machine translation
    from Portuguese to English.
    """
    def __init__(self):
        # Load the Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        # Initialize tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.
        """
        # Use SubwordTextEncoder to build tokenizers
        pt_sentences = []
        en_sentences = []
        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Build the tokenizers
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2**13)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2**13)

        return tokenizer_pt, tokenizer_en
