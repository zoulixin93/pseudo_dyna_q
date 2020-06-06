#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
import tensorflow as tf
from .base import basic_model
from .attention import vanilla_attention,vanilla_attention_for_all
from utils.time_analyse import timeit
from .base import create_optimizer

def products(x,y):
    return tf.reduce_sum(tf.multiply(x, tf.expand_dims(y, 0)), 2)


class rl_network(basic_model):
    def _create_placeholders(self):
        self.uid = tf.placeholder(tf.int32, (None,))
        self.iid = tf.placeholder(tf.int32, (None,))
        self.trajectory = tf.placeholder(tf.int32 ,(None,None))
        self.target_index = tf.placeholder(tf.int32,(None,2))
        self.feedback = tf.placeholder(tf.float32,(None,None))
        self.target_value = tf.placeholder(tf.float32, (None,))

    def _update_placehoders(self):
        self.placeholders["all"] = {"uid":self.uid,
                                    "iid":self.iid,
                                    "label":self.target_value,
                                    "trajectory":self.trajectory,
                                    "target_index":self.target_index,"feedback":self.feedback}
        predicts = ["uid","iid","trajectory","target_index","feedback"]
        predict_all = ["uid","trajectory","target_index","feedback"]
        self.placeholders["predict"]={item:self.placeholders["all"][item] for item in predicts}
        self.placeholders["optimizer"] = self.placeholders["all"]
        self.placeholders["predict_all"] = {item:self.placeholders["all"][item] for item in predict_all}

    def _create_inference(self):
        initializer = tf.random_uniform_initializer(0, 0.1, seed=self.config.RANDOM_SEED)
        item_feature = tf.Variable(tf.random_uniform([self.config.item_number, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='item_feature')
        user_feature = tf.Variable(tf.random_uniform([self.config.user_number, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='user_feature')
        projection_weight = [tf.Variable(tf.random_uniform([self.config.latent_factor, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='projections_'+str(i))
                             for i in range(self.config.feedback_number)]


        trajectory_embedding = tf.nn.embedding_lookup(item_feature,self.trajectory)
        for i in range(self.config.feedback_number):
            temp = tf.tensordot(trajectory_embedding,projection_weight[i],axes=[[2],[0]])
            mask = tf.cast(self.feedback-i*1.0,tf.bool)
            mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, self.config.latent_factor])
            trajectory_embedding = tf.where(mask,trajectory_embedding,temp)

        u_embedding = tf.nn.embedding_lookup(user_feature,self.uid)
        initial_state = tuple([u_embedding for _ in range(self.config.RNN_LAYER)])
        self.rnn_outputs,rnn_state = self.build_cell(self.config.CELL_TYPE,
                                                     initializer,
                                                     self.config.latent_factor,
                                                     trajectory_embedding,initial_state)
        rnn_out = tf.gather_nd(self.rnn_outputs, self.target_index)

        seq_mask = tf.cast(tf.transpose(self.trajectory,[1,0]),tf.bool)
        seq = tf.transpose(self.rnn_outputs,[1,0,2])
        s_atten = vanilla_attention(tf.nn.embedding_lookup(item_feature,self.iid),seq,seq_mask)
        w_last = [0.1,0.3,0.3]
        last_layer = [tf.Variable(tf.random_uniform((self.config.item_number,
                                                    self.config.latent_factor), 0.0, 0.1),
                                 name='last_layer', trainable=True) for i in range(3)]
        last_feature = [tf.nn.embedding_lookup(last_layer[i], self.iid) for i in range(3)]

        self.score = tf.reshape(tf.reduce_sum(w_last[0]*tf.multiply(u_embedding,last_feature[0]) \
                                              +w_last[1]*tf.multiply(rnn_out, last_feature[1])\
                                              +w_last[2]*tf.multiply(s_atten, last_feature[2]) ,axis=1),(-1,))
        self.f = tf.exp(self.score)
        expand_rnn = tf.reshape(tf.tile(rnn_out, [1, self.config.item_number]), (-1,self.config.item_number, self.config.latent_factor))
        expand_user = tf.reshape(tf.tile(u_embedding, [1, self.config.item_number]), (-1,self.config.item_number, self.config.latent_factor))
        # item_feature
        expand_atten = vanilla_attention_for_all(item_feature, seq, seq_mask) # batch,item,feature

        self.all_score = w_last[0]*products(expand_user,last_layer[0])+ \
                         w_last[1] * products(expand_rnn, last_layer[1]) + \
                         w_last[2] * products(expand_atten, last_layer[2])

    def _create_optimizer(self):
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score,labels=self.target_value))
        optimizer = create_optimizer(self.config.learning_rate, self.config.optimizer_name)
        gvs = optimizer.compute_gradients(self.loss)
        self.optimizer = optimizer.apply_gradients(gvs,global_step=self.global_step)

    def optimize_model(self, sess, data):
        feed_dicts = self._get_feed_dict("optimizer", data)
        return sess.run([self.loss, self.optimizer], feed_dicts)[0]

    def predict(self,sess,data):
        feed_dicts = self._get_feed_dict("predict", data)
        return sess.run(self.score, feed_dicts)

    def predict_f(self,sess,data):
        feed_dicts = self._get_feed_dict("predict", data)
        return sess.run(self.f, feed_dicts)

    def predict_all(self,sess,data):
        feed_dicts = self._get_feed_dict("predict_all", data)
        return sess.run(self.all_score, feed_dicts)


class q_network(rl_network):
    def _create_optimizer(self):
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.score-self.target_value))
        optimizer = create_optimizer(self.config.learning_rate, self.config.optimizer_name)
        gvs = optimizer.compute_gradients(self.loss)
        self.optimizer = optimizer.apply_gradients(gvs,global_step=self.global_step)
    pass
