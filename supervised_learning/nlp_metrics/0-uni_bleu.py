#!/usr/bin/env python3
"""This modlue containe the uni blue function"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list of list of str): List of reference translations.
        sentence (list of str): List containing the model proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    # Count the number of words in the sentence
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Count the maximum number of times each word appears in any reference
    max_counts = {}
    for reference in references:
        ref_counts = {}
        for word in reference:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        for word in ref_counts:
            if word in max_counts:
                max_counts[word] = max(max_counts[word], ref_counts[word])
            else:
                max_counts[word] = ref_counts[word]

    # Calculate the clipped counts
    clipped_counts = {}
    for word in word_counts:
        if word in max_counts:
            clipped_counts[word] = min(word_counts[word], max_counts[word])
        else:
            clipped_counts[word] = 0

    # Calculate precision
    precision = sum(clipped_counts.values()) / len(sentence)

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(
        ref_lengths, key=lambda ref_len: (
            abs(ref_len - len(sentence)), ref_len)
    )
    if len(sentence) > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / len(sentence))

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
