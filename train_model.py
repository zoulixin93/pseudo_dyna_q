#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
from arguments import parse_arguments,initialize
import utils
import os
import train
import time
from train import *
import tensorflow as tf
if __name__ == '__main__':
    np.random.seed(int(time.time()))
    config = []|parse_arguments|initialize
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_index)
    config.training_agent = model_training_with_weighted_regularizer
    training(config).run()


