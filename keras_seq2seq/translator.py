# -*- coding: utf-8 -*-
import time
import argparse
import attention_seq2seq as atn
import word2vec as w2v
import corpus_processor as cp
import config as cfg
from utils import get_logger

_logger = get_logger(__name__)


def reconstruct_nn(w2v_model):
    nn_params = {'nb_epoch': cfg.NN_EPOCH_NUM,
                 'input_dim': len(w2v_model.vocab),
                 'input_length': cfg.NN_SENTENCE_MAX_LENGTH_SOURCE,
                 'hidden_dim': cfg.NN_HIDDEN_DIM,
                 'output_length': cfg.NN_SENTENCE_MAX_LENGTH_TARGET,
                 'output_dim': len(w2v_model.vocab),
                 'depth': cfg.NN_DEPTH,
                 'corpus_name': cfg.NN_CORPUS_NAME}
    nn_model = atn.build_nn_model(nn_params)
    return nn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trading arguments')
    # parser.add_argument('-a', '--nn_architect_file', help='Neural network model file name')
    parser.add_argument('-w', '--nn_weight_file', help='Neural network model file name')
    parser.add_argument('-v', '--w2v_file', help='word2vec model file name')
    args = parser.parse_args()

    # if args.nn_architect_file is None or args.nn_weight_file is None or args.w2v_file is None:
    if args.nn_weight_file is None or args.w2v_file is None:
        print('Please specify model files.')
    else:
        print('Loading models...')
        ts = time.time()
        # nn_model = atn.load_model(args.nn_architect_file, args.nn_weight_file)
        w2v_model = w2v.load_model(args.w2v_file)
        nn_model = reconstruct_nn(w2v_model)
        atn.load_model(nn_model, args.nn_weight_file)
        print('Models loaded ({:.1f} minutes)'.format((time.time() - ts) / 60))
        while True:
            source = raw_input("EN> ")
            if source == '__EOT__':
                break
            target = nn_model.predict([cp.sentence_to_one_hot(source)])
            print('CH> ' + cp.one_hot_to_sentence(w2v_model, target))
