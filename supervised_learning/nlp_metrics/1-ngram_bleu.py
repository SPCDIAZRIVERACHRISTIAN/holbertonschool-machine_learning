#!/usr/bin/env python3
"""This modlue containe the uni blue ngram function"""
import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): List of reference translations.
        sentence (list of str): List containing the model proposed sentence.
        n (int): Size of the n-gram to use for evaluation.

    Returns:
        float: The n-gram BLEU score.
    """

    def ngrams(sequence, n):
        """Generate n-grams from a sequence of words.
        Args:
            sequence (list): List of words.
            n (int): Size of n-grams to generate.
        Returns:
            list: List of n-grams.
        """
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i: i + n])
            ngrams.append(ngram)
        return ngrams

    # Generate n-grams for the sentence
    sentence_ngrams = ngrams(sentence, n)
    sentence_ngram_counts = Counter(sentence_ngrams)

    # Generate n-grams for the references
    max_ngram_counts = Counter()
    for reference in references:
        reference_ngrams = ngrams(reference, n)
        reference_ngram_counts = Counter(reference_ngrams)
        for ngram in reference_ngram_counts:
            max_ngram_counts[ngram] = max(
                max_ngram_counts[ngram], reference_ngram_counts[ngram]
            )

    # Calculate clipped counts
    clipped_ngram_counts = {
        ngram: min(count, max_ngram_counts[ngram])
        for ngram, count in sentence_ngram_counts.items()
    }

    # Calculate precision
    precision = sum(
        clipped_ngram_counts.values()) / max(
            1, len(sentence_ngrams))

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
