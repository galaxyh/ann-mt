# -*- coding: utf-8 -*-

import time
import word2vec
import corpus_processor as cp
import config as cfg

from keras.models import Sequential
from keras.layers.core import Activation
from keras.models import model_from_json
from seq2seq.models import AttentionSeq2seq
from utils import get_logger

_logger = get_logger(__name__)


def build_nn_model(nn_params):
    ts = time.time()
    _logger.info('Building NN model...')
    model = Sequential()
    model.add(AttentionSeq2seq(input_dim=nn_params['input_dim'],
                               input_length=nn_params['input_length'],
                               hidden_dim=nn_params['hidden_dim'],
                               output_length=nn_params['output_length'],
                               output_dim=nn_params['output_dim'],
                               depth=nn_params['depth']))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    save_model_architecture(model, nn_params)
    _logger.info('Done building NN model ({:.1f} minutes).'.format((time.time() - ts) / 60))
    return model


def train(nn_model, batch_iter):
    _logger.info('Training...')
    batch_count = 0
    for array_s, array_t in batch_iter:
        nn_model.fit(array_s, array_t, nb_epoch=1, batch_size=len(array_s), show_accuracy=True)
        batch_count += 1
        if batch_count % 10 == 0:
            _logger.info('Processed sentences {}'.format(batch_count * len(array_s)))


def test(nn_model, test_batch):
    ts = time.time()
    _logger.info('Evaluating...')
    objective_score = nn_model.evaluate(test_batch)
    _logger.info('Objective score = {}'.format(objective_score))
    _logger.info('Done evaluation ({:.1f} minutes)'.format((time.time() - ts) / 60))


def gen_nn_model_name(nn_params):
    model_name = nn_params['corpus_name']
    model_name += '_e' + str(nn_params['nb_epoch'])
    model_name += '_id' + str(nn_params['input_dim'])
    model_name += '_il' + str(nn_params['input_length'])
    model_name += '_hd' + str(nn_params['hidden_dim'])
    model_name += '_ol' + str(nn_params['output_length'])
    model_name += '_od' + str(nn_params['output_dim'])
    model_name += '_d' + str(nn_params['depth'])
    return model_name


def save_model_architecture(nn_model, nn_params):
    model_name = gen_nn_model_name(nn_params)
    model_path_architecture = cfg.NN_MODEL_DIR + '/' + model_name + '.json'
    json_string = nn_model.to_json()
    open(model_path_architecture, 'w').write(json_string)


def save_model_weights(nn_model, nn_params):
    model_name = gen_nn_model_name(nn_params)
    model_path_weights = cfg.NN_MODEL_DIR + '/' + model_name + '.h5'
    nn_model.save_weights(model_path_weights, overwrite=True)


def load_model(architecture_filename, weights_filename):
    ts = time.time()
    _logger.info('Loading ANN model...')
    nn_model = model_from_json(open(architecture_filename).read())
    nn_model.load_weights(weights_filename)
    _logger.info('ANN Model loaded ({:.1f} minutes).'.format((time.time() - ts) / 60))
    return nn_model


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

    nn_params = {'nb_epoch': cfg.NN_EPOCH_NUM,
                 'input_dim': len(w2v_model.vocab),
                 'input_length': cfg.NN_SENTENCE_MAX_LENGTH_SOURCE,
                 'hidden_dim': cfg.NN_HIDDEN_DIM,
                 'output_length': cfg.NN_SENTENCE_MAX_LENGTH_TARGET,
                 'output_dim': len(w2v_model.vocab),
                 'depth': cfg.NN_DEPTH,
                 'corpus_name': cfg.NN_CORPUS_NAME}
    nn_model = build_nn_model(nn_params)

    ts0 = time.time()
    for full_data_pass_num in xrange(1, cfg.NN_EPOCH_NUM + 1):
        _logger.info('Full-data-pass iteration num: ' + str(full_data_pass_num))
        one_hot_iter = cp.get_parallel_one_hot_ndarray_iter(w2v_model.vocab,
                                                            cfg.NN_BATCH_SIZE,
                                                            corpus_train_source,
                                                            corpus_train_target,
                                                            maxlen_a=maxlen_source,
                                                            maxlen_b=maxlen_target)
        train(nn_model, one_hot_iter)
        save_model_weights(nn_model, nn_params)
        _logger.info(
            'Average time per full-pass: {:.1f} minutes.'.format((time.time() - ts0) / full_data_pass_num / 60))
    _logger.info('Done training ({:.1f} minutes)'.format((time.time() - ts0) / 60))

    '''_logger.info('Prediction and ground truth:')
    for p, g in zip(nn_model.predict(x_test), y_test):
        _logger.info('[P] {}'.format(one_hot_to_sentence(w2v_model, p)))
        _logger.info('[G] {}'.format(one_hot_to_sentence(w2v_model, g)))'''
