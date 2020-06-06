#!/usr/bin/python
# encoding: utf-8


import numpy as np

import ipdb
import os
import tensorflow as tf

def save_model(saver, path, sess, global_step):
    temp = path[:str(path).rfind('/')]
    os.system("rm " + temp + '/*')
    if not os.path.isdir(temp):
        os.mkdir(temp)
    saver.save(sess, path, global_step)

def create_saved_path(index,save_root_path,save_sub_path):
    return os.path.join(os.path.join(save_root_path,
                        save_sub_path+"_"+str(index)),
                        save_sub_path+"_"+str(index))

def update_target_graph_ops(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def load_model(saver,sess,path):
    saver.restore(sess, tf.train.latest_checkpoint(path))
    pass