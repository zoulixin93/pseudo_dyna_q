#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
import tensorflow as tf

def vanilla_attention(item,seq,seq_mask,**args):
    """
    :param item:[batch,embedding]
    :param seq:[batch,seq_length,embedding]
    :param seq_mask:[batch,seq_length]
    :return:[batch,embedding]
    """
    seq_mask = tf.expand_dims(seq_mask, axis=1) # s*1*seq_length
    item = tf.expand_dims(item, axis=1)  # B*1*H
    sim_matrix = tf.matmul(item, tf.transpose(seq, [0, 2, 1]))  # B*1*L
    padding = tf.ones_like(sim_matrix) * (-2 ** 32 + 1)
    atten = tf.where(seq_mask, sim_matrix, padding)
    score = tf.nn.softmax(atten) # B*1*L
    attention = tf.reshape(tf.matmul(score,seq), (tf.shape(atten)[0],-1))
    return attention


def vanilla_attention_for_all(item, seq, seq_mask, **args):
    """
    :param item: [item_number,embedding]
    :param seq: [1,seq_length,embedding]
    :param seq_mask: [1,seq_length]
    :return: [item_number,seq_length]
    """
    sim_matrix = tf.transpose(tf.tensordot(seq,item,axes=[[2],[1]]),[0,2,1])
    seq_mask = tf.reshape(tf.tile(seq_mask, [1,tf.shape(item)[0]]), (tf.shape(seq)[0], tf.shape(item)[0], -1))
    padding = tf.ones_like(sim_matrix) * (-2 ** 32 + 1)
    atten = tf.expand_dims(tf.nn.softmax(tf.where(seq_mask, sim_matrix, padding), 2),-1)
    seq = tf.tile(tf.expand_dims(seq,1),(1,tf.shape(item)[0],1,1))
    context = tf.reduce_sum(tf.multiply(atten,seq),2)
    return context