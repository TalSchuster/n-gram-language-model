#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    for sent in dataset:
        for i in xrange(len(sent)):
            uni = sent[i]
            if uni not in unigram_counts:
                unigram_counts[uni] = 0

            unigram_counts[uni] += 1

            if i >= 1:
                bi = (sent[i-1], sent[i])
                if bi not in bigram_counts:
                    bigram_counts[bi] = 0

                bigram_counts[bi] += 1

            if i >= 2:
                tri = (sent[i-2], sent[i-1], sent[i])
                if tri not in trigram_counts:
                    trigram_counts[tri] = 0

                trigram_counts[tri] += 1

            token_count += 1

    return trigram_counts, bigram_counts, unigram_counts, token_count


def get_unigram_probabilty(word_index, unigram_counts, train_token_count):
    if word_index in unigram_counts:
        return np.float64(unigram_counts[word_index])/train_token_count
    return np.float64(0)

def get_bigram_probabilty(first_word_index, second_word_index, bigram_counts, unigram_counts):
    two_tuple = (first_word_index, second_word_index)
    if two_tuple in bigram_counts:
        return np.float64(bigram_counts[two_tuple])/unigram_counts[first_word_index]
    return np.float64(0)

def get_trigram_probabilty(first_word_index, second_word_index, third_word_index, trigram_counts, bigram_counts):
    three_tuple = (first_word_index, second_word_index, third_word_index)
    two_tuple = (first_word_index, second_word_index)
    if three_tuple in trigram_counts:
        return np.float64(trigram_counts[three_tuple])/bigram_counts[two_tuple]
    return np.float64(0)

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    for sentence in eval_dataset:
        prob_sentence = np.float64(1)
        for word_num in xrange(2, len(sentence)):
            # Calculating the three probabilities (unigram, bigram and trigram)
            prob_unigram = get_unigram_probabilty(sentence[word_num], unigram_counts, train_token_count)
            prob_bigram = get_bigram_probabilty(sentence[word_num-1], sentence[word_num], bigram_counts, unigram_counts)
            prob_trigram = get_trigram_probabilty(sentence[word_num-2], sentence[word_num-1], sentence[word_num], trigram_counts, bigram_counts)

            # Calculating the prob. of this word to be the next word in the
            # sentence.
            prob_word = lambda1 * prob_unigram + lambda2 * prob_bigram + (1 - lambda1 - lambda2) * prob_trigram
            prob_sentence *= prob_word
        perplexity += np.log2(prob_sentence)
    perplexity /= len(eval_dataset)
    perplexity = np.power(2, -perplexity)
    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)

    # Lambda grid search
    min_perp = perplexity
    min_lambda = (0.5,0.4)
    for lambda1 in np.arange(0.0, 1.0, 0.1):
        for lambda2 in np.arange(0.0, 1.0 - lambda1, 0.1):
            perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)

            print 'lamda1: %.2f lambda2: %.2f lambda3: %.2f perplexity: %s' % (lambda1, lambda2, 1.0 - lambda1 - lambda2, perplexity)
            if perplexity < min_perp:
                min_perp = perplexity
                min_lambda = (lambda1, lambda2)

    perplexity = min_perp
    lambda1, lambda2 = min_lambda
    print '## Minimum perplexity resluts:'
    print 'lamda1: %.2f lambda2: %.2f lambda3: %.2f perplexity: %s' % (lambda1, lambda2, 1.0 - lambda1 - lambda2, perplexity)


if __name__ == "__main__":
    test_ngram()
