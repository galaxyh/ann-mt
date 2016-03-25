# -*- coding: utf-8 -*-
import time
import argparse
import attention_seq2seq as atn
import word2vec as w2v
import corpus_processor as cp
from utils import get_logger

_logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trading arguments')
    parser.add_argument('-a', '--nn_architect_file', help='Neural network model file name')
    parser.add_argument('-w', '--nn_weight_file', help='Neural network model file name')
    parser.add_argument('-v', '--w2v_file', help='word2vec model file name')
    args = parser.parse_args()

    if args.nn_architect_file is None or args.nn_weight_file is None or args.w2v_file is None:
        print('Please specify model files.')
    else:
        print('Loading models...')
        ts = time.time()
        nn_model = atn.load_model(args.nn_architect_file, args.nn_weight_file)
        w2v_model = w2v.load_model(args.w2v_file)
        print('Models loaded ({:.1f} minutes)'.format((time.time() - ts) / 60))
        while True:
            source = raw_input("EN> ")
            if source == '__EOT__':
                break
            target = nn_model.predict([cp.sentence_to_one_hot(source)])
            print('CH> ' + cp.one_hot_to_sentence(w2v_model, target))
