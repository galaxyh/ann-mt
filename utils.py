# -*- coding: utf-8 -*-

import logging


def get_logger(name):
    logging.basicConfig(filename=(name + '.log'),
                        level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s')
    return logging.getLogger(name)


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time
