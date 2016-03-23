# -*- coding: utf-8 -*-

import codecs
import numpy as np

from config import EOS_SYMBOL, UNK_SYMBOL
from utils import get_logger

_logger = get_logger(__name__)

MAX_INT_VALUE = 9223372036854775807


class IterableSentences(object):
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        for line in codecs.open(self._filename, 'r', 'utf-8'):
            yield line.strip()


def tokenize(text):
    return text.split()


def get_sentences(corpus):
    if not isinstance(corpus, list):
        corpus_sources = [corpus]
    else:
        corpus_sources = corpus

    for c in corpus_sources:
        iterable_corpus_lines = IterableSentences(c)
        for line in iterable_corpus_lines:
            tokens = tokenize(line)
            tokens.append(EOS_SYMBOL)
            yield tokens


def get_parallel_sentences_iter(corpus_a, corpus_b, maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
    for (sent_a, sent_b) in zip(get_sentences(corpus_a), get_sentences(corpus_b)):
        if len(sent_a) <= maxlen_a and len(sent_b) <= maxlen_b:
            yield sent_a, sent_b


def get_parallel_sentences_all(corpus_a, corpus_b, maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
    all_a = []
    all_b = []
    for (sent_a, sent_b) in get_parallel_sentences_iter(corpus_a, corpus_b, maxlen_a=maxlen_a, maxlen_b=maxlen_b):
        all_a.append(sent_a)
        all_b.append(sent_b)
    return all_a, all_b


def get_processed_sentences(vocab, corpus):
    for line in get_sentences(corpus):
        yield [UNK_SYMBOL if w not in vocab else w for w in line]


def get_parallel_processed_sentences_iter(vocab, corpus_a, corpus_b, maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
    for (sent_a, sent_b) in zip(get_processed_sentences(vocab, corpus_a),
                                get_processed_sentences(vocab, corpus_b)):
        if len(sent_a) <= maxlen_a and len(sent_b) <= maxlen_b:
            yield sent_a, sent_b


def get_parallel_processed_sentences_all(parallel_sentence_iter):
    all_a = []
    all_b = []
    for (sent_a, sent_b) in parallel_sentence_iter:
        all_a.append(sent_a)
        all_b.append(sent_b)
    return all_a, all_b


def _get_parallel_batch(parallel_sentences, batch_size=1):
    batch = []
    for s_a, s_b in parallel_sentences:
        batch.append((s_a, s_b))
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield []


def get_parallel_processed_sentences_batch_iter(parallel_sentence_iter, batch_size=1):
    for parallel_batch in _get_parallel_batch(parallel_sentence_iter, batch_size):
        batch_a = []
        batch_b = []
        if parallel_batch:
            for pb in parallel_batch:
                batch_a.append(pb[0])
                batch_b.append(pb[1])
        yield batch_a, batch_b


def word_to_one_hot(vocab, word):
    v = np.zeros(len(vocab))
    v[vocab[word].index] = 1.0
    return v


def one_hot_to_word(index2word, word_vec):
    return index2word[np.argmax(word_vec)]


def sentence_to_one_hot(maxlen, vocab, sentence):
    s = np.full((maxlen, len(vocab)), word_to_one_hot(vocab, EOS_SYMBOL))
    for i, t in enumerate(sentence):
        s[i] = word_to_one_hot(vocab, t)
    return s


def one_hot_to_sentence(index2word, sentence_vec):
    return [one_hot_to_word(index2word, wv) for wv in sentence_vec]


def get_one_hot_ndarray(sentences, max_sent_len, vocab):
    out_ndarray = np.empty((len(sentences), max_sent_len, len(vocab)))
    for i, s in enumerate(sentences):
        out_ndarray[i] = sentence_to_one_hot(max_sent_len, vocab, s)
    return out_ndarray


def get_parallel_one_hot_ndarray_iter(vocab, batch_size,
                                      corpus_a, corpus_b,
                                      maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
    parallel_iter = get_parallel_processed_sentences_iter(vocab,
                                                          corpus_a, corpus_b,
                                                          maxlen_a=maxlen_a, maxlen_b=maxlen_b)
    batch_iter = get_parallel_processed_sentences_batch_iter(parallel_iter, batch_size=batch_size)
    for batch_s, batch_t in batch_iter:
        array_s = get_one_hot_ndarray(batch_s, maxlen_a, vocab)
        array_t = get_one_hot_ndarray(batch_t, maxlen_b, vocab)
        yield array_s, array_t
