# -*- coding: utf-8 -*-

import codecs

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


def get_parallel_sentences(corpus_a, corpus_b, maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
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


def get_parallel_processed_sentences(vocab, corpus_a, corpus_b, maxlen_a=MAX_INT_VALUE, maxlen_b=MAX_INT_VALUE):
    all_a = []
    all_b = []
    for (sent_a, sent_b) in get_parallel_processed_sentences_iter(vocab,
                                                                  corpus_a, corpus_b,
                                                                  maxlen_a=maxlen_a, maxlen_b=maxlen_b):
        all_a.append(sent_a)
        all_b.append(sent_b)
    return all_a, all_b
