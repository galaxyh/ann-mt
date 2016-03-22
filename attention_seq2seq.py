# -*- coding: utf-8 -*-

import time
import numpy as np
import word2vec
import corpus_processor as cp
import config as cfg

from keras.layers.core import Activation
from seq2seq.models import AttentionSeq2seq
from utils import get_logger

_logger = get_logger(__name__)


def word_to_one_hot(w2vm, word):
    v = np.zeros(len(w2vm.vocab))
    v[w2vm.vocab[word].index] = 1.0
    return v


def one_hot_to_word(w2vm, word_vec):
    return w2vm.index2word[np.argmax(word_vec)]


def sentence_to_one_hot(maxlen, w2vm, sentence):
    s = np.full((maxlen, len(w2v_model.vocab)), word_to_one_hot(w2vm, cfg.EOS_SYMBOL))
    for i, t in enumerate(sentence):
        s[i] = word_to_one_hot(w2vm, t)
    return s


def one_hot_to_sentence(w2vm, sentence_vec):
    return [one_hot_to_word(w2vm, wv) for wv in sentence_vec]


def get_one_hot_ndarray(sentences, max_sent_len, w2vm):
    out_ndarray = np.empty((len(sentences), max_sent_len, len(w2vm.vocab)))
    for i, s in enumerate(sentences):
        out_ndarray[i] = sentence_to_one_hot(max_sent_len, w2vm, s)
    return out_ndarray


def train_test(nn_params, train_pair, test_pair):
    (x_train, y_train) = train_pair
    (x_test, y_test) = test_pair

    _logger.info('x_train shape: {}'.format(x_train.shape))
    _logger.info('y_train shape: {}'.format(y_train.shape))
    _logger.info('x_test shape: {}'.format(x_test.shape))
    _logger.info('y_test shape: {}'.format(y_test.shape))

    ts0 = time.time()
    ts1 = time.time()

    _logger.info('Building NN model...')
    model = AttentionSeq2seq(input_dim=nn_params['input_dim'],
                             input_length=nn_params['input_length'],
                             hidden_dim=nn_params['hidden_dim'],
                             output_length=nn_params['output_length'],
                             output_dim=nn_params['output_dim'],
                             depth=nn_params['depth'])
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    _logger.info('Done building NN model ({:.1f} minutes).'.format((time.time() - ts1) / 60))

    ts1 = time.time()

    _logger.info('Training...')
    model.fit(x_train, y_train, nb_epoch=nn_params['nb_epoch'], show_accuracy=True)
    _logger.info('Done training ({:.1f} minutes).'.format((time.time() - ts1) / 60))

    ts1 = time.time()

    _logger.info('Evaluating...')
    objective_score = model.evaluate(x_test, y_test)
    _logger.info('Objective score = {}'.format(objective_score))
    _logger.info('Done evaluation ({:.1f} minutes)'.format((time.time() - ts1) / 60))

    _logger.info('Total time elapsed: {:.1f} minutes.'.format((time.time() - ts0) / 60))

    return model


def save_model(nn_model, nn_params, corpus_name):
    model_name = corpus_name
    model_name += '_e' + str(nn_params['nb_epoch'])
    model_name += '_id' + str(nn_params['input_dim'])
    model_name += '_il' + str(nn_params['input_length'])
    model_name += '_hd' + str(nn_params['hidden_dim'])
    model_name += '_ol' + str(nn_params['output_length'])
    model_name += '_ld' + str(nn_params['output_dim'])
    model_name += '_d' + str(nn_params['depth'])
    model_path = cfg.NN_MODEL_DIR + '/' + model_name
    nn_model.save_weights(model_path, overwrite=True)


if __name__ == '__main__':
    corpus_train_source = cfg.TRAIN_CORPUS_SOURCE
    corpus_train_target = cfg.TRAIN_CORPUS_TARGET
    corpus_test_source = cfg.TEST_CORPUS_SOURCE
    corpus_test_target = cfg.TEST_CORPUS_TARGET
    maxlen_source = cfg.NN_SENTENCE_MAX_LENGTH_SOURCE
    maxlen_target = cfg.NN_SENTENCE_MAX_LENGTH_TARGET

    sentences_all = cp.get_sentences([corpus_train_source, corpus_train_target])

    w2v_params = {'win_size': cfg.W2V_WINDOW_SIZE,
                  'min_w_num': cfg.W2V_TOKEN_MIN_FREQUENCY,
                  'vect_size': cfg.W2V_VECTOR_SIZE,
                  'workers_num': cfg.W2V_WORKER_NUM,
                  'corpus_name': cfg.W2V_CORPUS_NAME,
                  'new_models_dir': cfg.W2V_MODEL_DIR}

    _logger.info('Building word2vec model...')
    ts0 = time.time()
    w2v_model = word2vec.get_model(w2v_params, sentences_all, rebuild=cfg.W2V_REBUILD_MODEL)
    _logger.info('Done building word2vec model ({:.1f} minutes)'.format((time.time() - ts0) / 60))

    _logger.info('Preparing train/test data...')
    ts0 = time.time()
    sent_train_s, sent_train_t = cp.get_parallel_processed_sentences(w2v_model.vocab,
                                                                     corpus_train_source,
                                                                     corpus_train_target,
                                                                     maxlen_a=maxlen_source,
                                                                     maxlen_b=maxlen_target)

    x_train = get_one_hot_ndarray(sent_train_s, maxlen_source, w2v_model)
    y_train = get_one_hot_ndarray(sent_train_t, maxlen_target, w2v_model)

    sent_test_s, sent_test_t = cp.get_parallel_processed_sentences(w2v_model.vocab,
                                                                   corpus_test_source,
                                                                   corpus_test_target,
                                                                   maxlen_a=maxlen_source,
                                                                   maxlen_b=maxlen_target)

    x_test = get_one_hot_ndarray(sent_test_s, maxlen_source, w2v_model)
    y_test = get_one_hot_ndarray(sent_test_t, maxlen_target, w2v_model)
    _logger.info('Done preparing train/test data ({:.1f} minutes)'.format((time.time() - ts0) / 60))

    nn_param = {'nb_epoch': cfg.NN_EPOCH_NUM,
                'input_dim': x_train.shape[2],
                'input_length': x_train.shape[1],
                'hidden_dim': cfg.NN_HIDDEN_DIM,
                'output_length': y_train.shape[1],
                'output_dim': y_train.shape[2],
                'depth': cfg.NN_DEPTH}

    attention_model = train_test(nn_param, (x_train, y_train), (x_test, y_test))
    save_model(attention_model)

    _logger.info('Prediction and ground truth:')
    for p, g in zip(attention_model.predict(x_test), y_test):
        _logger.info('[P] {}'.format(one_hot_to_sentence(w2v_model, p)))
        _logger.info('[G] {}'.format(one_hot_to_sentence(w2v_model, g)))
