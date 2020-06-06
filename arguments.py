#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
import argparse
import utils
import os
import tensorflow as tf
# adding breakpoint with ipdb.set_trace()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@utils.Pipe
def parse_arguments(*args):
    parser = argparse.ArgumentParser(description='robust model based reinforcement learning for recommendation')
    parser.add_argument('-log_path',dest='log_path',type = str,default="./log/")
    parser.add_argument('-model',dest='model',type = str,default="random")
    parser.add_argument('-GPU_index',dest='GPU_index',type=str,default="0")
    parser.add_argument('-u_file',dest='u_file',type=str,default="./data/retailrocket/u_factor")
    parser.add_argument('-i_file',dest='i_file',type=str,default="./data/retailrocket/i_factor")
    parser.add_argument('-entropy_file',dest='entropy_file',type=str,default="./data/retailrocket/entropy")
    parser.add_argument('-user_number',dest='user_number',type=int,default=100)
    parser.add_argument('-item_number', dest='item_number', type=int, default=200)
    parser.add_argument('-latent_factor', dest='latent_factor', type=int, default=20)
    parser.add_argument('-feedback_number', dest='feedback_number', type=int, default=2)
    parser.add_argument('-batch_size',dest='batch_size',type=int,default=128)
    parser.add_argument('-buffer_size',dest='buffer_size',type=int,default=5000)
    parser.add_argument("-learning_rate",dest='learning_rate',type=float,default=0.1)
    parser.add_argument("-trajectory_number",dest='trajectory_number',type=int,default=10000)
    parser.add_argument("-epoch",dest='epoch',type=int,default=10000)
    parser.add_argument("-evaluate_round",dest='evaluate_round',type=int,default=1000)
    parser.add_argument("-gamma",dest='gamma',type=float,default=0.9)
    parser.add_argument("-RNN_LAYER",dest='RNN_LAYER',type=int,default=1)
    parser.add_argument("-RANDOM_SEED",dest='RANDOM_SEED',type=int,default=123)
    parser.add_argument("-CELL_TYPE",dest='CELL_TYPE',type=str,default="nlstm")
    parser.add_argument("-temperature",dest="temperature",type=float,default=100)
    parser.add_argument("-evaluation_num",dest="evaluation_num",type=int,default=100)
    parser.add_argument("-maximum_weight",dest="maximum_weight",type=float,default=10)
    parser.add_argument("-regularizer_weight",dest="regularizer_weight",type=float,default=0.01)
    parser.add_argument("-optimizer_name",dest='optimizer_name',type=str,default="sgd")
    parser.add_argument("-threshold",dest='threshold',type=float,default=0.125)
    parser.add_argument("-save_path",dest='save_path',type=str,default="./data/temp_sample")
    parser.add_argument("-sample_n",dest='sample_n',type=int,default=10)
    parser.add_argument("-mini_temperature",dest='mini_temperature',type=int,default=5.0)
    args = parser.parse_args()
    args.GPU_OPTION = tf.GPUOptions(allow_growth=True)
    args.RANDOM_SEED = 123
    args.LATENT_FACTOR = 200
    args.RNN_HIDDEN = 200
    args.RNN_LAYER = 1
    return args

@utils.Pipe
def initialize(config,*args):
    if not os.path.isdir(config.log_path):
        os.mkdir(config.log_path)
    utils.log.set_log_path(os.path.join(config.log_path,config.model)+"_"+str(utils.get_now_time())+".log")
    utils.log.info('saving log file in '+utils.log().log_path)
    utils.log.structure_info("config for experiments",list(vars(config).items()))
    return config