#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
from .rl_network import rl_network
import tensorflow as tf
from .attention import vanilla_attention_for_all,vanilla_attention
from .base import create_optimizer

class reward_network(rl_network):
    c_threshold = 0.0
    t_threshold = 0.0
    def _create_placeholders(self):
        self.uid = tf.placeholder(tf.int32, (None,))
        self.iid = tf.placeholder(tf.int32, (None,))
        self.trajectory = tf.placeholder(tf.int32 ,(None,None))
        self.target_index = tf.placeholder(tf.int32,(None,2))
        self.feedback = tf.placeholder(tf.float32,(None,None))
        self.terminate = tf.placeholder(tf.float32,(None,))
        self.target_value = tf.placeholder(tf.float32, (None,))
        self.weight = tf.placeholder(tf.float32,(None,))

    def _update_placehoders(self):
        self.placeholders["all"] = {"uid":self.uid,
                                    "iid":self.iid,
                                    "label":self.target_value,
                                    "trajectory":self.trajectory,
                                    "target_index":self.target_index,
                                    "feedback":self.feedback,
                                    "terminate":self.terminate,
                                    "weight":self.weight}
        predicts = ["uid","iid","trajectory","target_index","feedback"]
        predict_all = ["uid","trajectory","target_index","feedback"]
        self.placeholders["predict"]={item:self.placeholders["all"][item] for item in predicts}
        self.placeholders["optimizer"] = self.placeholders["all"]
        self.placeholders["predict_all"] = {item:self.placeholders["all"][item] for item in predict_all}

    def _create_inference(self):
        initializer = tf.random_uniform_initializer(0, 0.1, seed=self.config.RANDOM_SEED)
        item_feature = tf.Variable(tf.random_uniform([self.config.item_number, self.config.latent_factor],
                                                     0, 0.1), trainable=self.trainable, name='item_feature')
        user_feature = tf.Variable(tf.random_uniform([self.config.user_number, self.config.latent_factor],
                                                     0, 0.1), trainable=self.trainable, name='user_feature')
        projection_weight = [tf.Variable(tf.random_uniform([self.config.latent_factor, self.config.latent_factor],
                                                           0, 0.1), trainable=self.trainable,
                                         name='projections_' + str(i))
                             for i in range(self.config.feedback_number)]

        ##### item feature embedding
        trajectory_embedding = tf.nn.embedding_lookup(item_feature, self.trajectory)
        for i in range(self.config.feedback_number):
            temp = tf.tensordot(trajectory_embedding, projection_weight[i], axes=[[2], [0]])
            mask = tf.cast(self.feedback - i * 1.0, tf.bool)
            mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, self.config.latent_factor])
            trajectory_embedding = tf.where(mask, trajectory_embedding, temp)

        ##### rnn processing
        u_embedding = tf.nn.embedding_lookup(user_feature, self.uid)
        initial_state = tuple([u_embedding for _ in range(self.config.RNN_LAYER)])
        self.rnn_outputs, rnn_state = self.build_cell(self.config.CELL_TYPE,
                                                      initializer,
                                                      self.config.latent_factor,
                                                      trajectory_embedding, initial_state)
        rnn_out = tf.gather_nd(self.rnn_outputs, self.target_index)

        #### attentive
        seq_mask = tf.cast(tf.transpose(self.trajectory, [1, 0]), tf.bool)
        seq = tf.transpose(self.rnn_outputs, [1, 0, 2])
        s_atten = vanilla_attention(tf.nn.embedding_lookup(item_feature, self.iid), seq, seq_mask)

        #### calculate the score
        w_last = [0.3,0.3,0.3]
        last_layer = [tf.Variable(tf.random_uniform((self.config.item_number,
                                                     self.config.latent_factor), 0.0, 0.1),
                                  name='last_layer', trainable=True) for i in range(3)]
        last_feature = [tf.nn.embedding_lookup(last_layer[i], self.iid) for i in range(3)]

        self.score = tf.reshape(tf.reduce_sum(w_last[0] * tf.multiply(u_embedding, last_feature[0]) \
                                              + w_last[1] * tf.multiply(rnn_out, last_feature[1]) \
                                              + w_last[2] * tf.multiply(s_atten, last_feature[2]), axis=1), (-1,))

        expand_rnn = tf.reshape(tf.tile(rnn_out, [1, self.config.item_number]), (self.config.item_number, -1))
        expand_user = tf.reshape(tf.tile(u_embedding, [1, self.config.item_number]), (self.config.item_number, -1))
        expand_atten = vanilla_attention_for_all(item_feature, seq, seq_mask)

        self.all_score = tf.reduce_sum(
                        w_last[0] * tf.multiply(expand_user, last_layer[0]) + \
                        w_last[1] * tf.multiply(expand_rnn, last_layer[1]) + \
                        w_last[2] * tf.multiply(expand_atten, last_layer[2]), 1)

        #### terminate
        w_terminate = [0.3, 0.3, 0.3]
        terminate_weight = [tf.Variable(tf.random_uniform((self.config.item_number,
                                                     self.config.latent_factor), 0.0, 0.1),
                                  name='last_layer', trainable=True) for i in range(3)]
        terminate_feature = [tf.nn.embedding_lookup(terminate_weight[i], self.iid) for i in range(3)]
        self.t_score = tf.reshape(tf.reduce_sum(w_terminate[0] * tf.multiply(u_embedding, terminate_feature[0]) \
                                              + w_terminate[1] * tf.multiply(rnn_out, terminate_feature[1]) \
                                              + w_terminate[2] * tf.multiply(s_atten, terminate_feature[2]), axis=1), (-1,))
        self.t_p = tf.sigmoid(self.t_score)
        self.f = tf.sigmoid(self.score)
        self.a_f = tf.sigmoid(self.all_score)

    def _create_optimizer(self):
        raw_loss =tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score,labels=self.target_value)+\
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.t_score,labels=self.terminate)
        self.raw_loss = tf.reduce_sum(tf.multiply(self.weight,raw_loss))
        self.regularizer1 = self.config.regularizer_weight*(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score,labels=tf.ones_like(self.score))+
                                                            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.t_score,labels=tf.zeros_like(self.t_score)))
        self.loss = tf.reduce_sum(tf.multiply(self.weight,raw_loss))+tf.reduce_sum(self.regularizer1)
        optimizer = create_optimizer(self.config.learning_rate*10,self.config.optimizer_name)
        gvs = optimizer.compute_gradients(self.loss)
        self.optimizer = optimizer.apply_gradients(gvs,global_step=self.global_step)
        self.loss_neg = tf.reduce_sum(self.config.regularizer_weight*(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score,labels=tf.zeros_like(self.score))+
                                                                      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.t_score,labels=tf.ones_like(self.t_score))))
        optimizer_neg = create_optimizer(self.config.learning_rate*10,self.config.optimizer_name)
        gvs_neg = optimizer_neg.compute_gradients(self.loss_neg)
        self.optimizer_neg = optimizer_neg.apply_gradients(gvs_neg,global_step=self.global_step)

    def optimize_model_click(self, sess, data):
        feed_dicts = self._get_feed_dict("optimizer", data)
        return sess.run([self.raw_loss,self.optimizer], feed_dicts)[0]

    def optimize_model_neg(self, sess, data):
        feed_dicts = self._get_feed_dict("optimizer", data)
        return sess.run([self.loss_neg, self.optimizer_neg], feed_dicts)[0]

    def predict(self,sess,data):
        feed_dicts = self._get_feed_dict("predict", data)
        t,p_c = sess.run([self.t_p,self.f], feed_dicts)
        return p_c,t

    def predict_all(self,sess,data):
        feed_dicts = self._get_feed_dict("predict_all", data)
        return sess.run([self.t_p,self.a_f], feed_dicts)







