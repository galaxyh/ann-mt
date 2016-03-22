# -*- coding: utf-8 -*-

TRAIN_CORPUS_SOURCE = 'corpus/corpus.train.source.txt'
TRAIN_CORPUS_TARGET = 'corpus/corpus.train.target.txt'
TEST_CORPUS_SOURCE = 'corpus/corpus.test.source.txt'
TEST_CORPUS_TARGET = 'corpus/corpus.test_target.txt'

EOS_SYMBOL = 'E#O#S'
UNK_SYMBOL = 'U#N#K'

W2V_WINDOW_SIZE = 5
W2V_TOKEN_MIN_FREQUENCY = 2
W2V_VECTOR_SIZE = 500
W2V_WORKER_NUM = 10
W2V_CORPUS_NAME = '20K'
W2V_MODEL_DIR = 'w2v_model'

NN_SENTENCE_MAX_LENGTH_SOURCE = 30
NN_SENTENCE_MAX_LENGTH_TARGET = 30
NN_MODEL_DIR = 'nn_model'

NN_EPOCH_NUM = 20
NN_HIDDEN_DIM = 600
NN_DEPTH = 4
