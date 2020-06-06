#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
import tensorflow as tf
from utils.zlog import log

import numpy as np
import ipdb
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.contrib.model_pruning.python.layers import core_layers
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_clipped_value = 100000000.0
class nlstm(tf.nn.rnn_cell.BasicLSTMCell):
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def build(self, inputs_shape):
        print("ntlstm cell")
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value-1
        h_depth = self._num_units
        # input gate
        self.W_xi = self.add_variable("w_xi",shape=(input_depth, h_depth))
        self.W_hi = self.add_variable("w_hi",shape=(h_depth, h_depth))
        self.bias_i = self.add_variable("bias_i",shape=(h_depth,))
        self.w_ci = self.add_variable("w_ci",shape=(h_depth,))
        # forget gate
        self.W_xf = self.add_variable("w_xf",shape=(input_depth, h_depth))
        self.W_hf = self.add_variable("w_hf",shape=(h_depth, h_depth))
        self.w_cf = self.add_variable("w_cf",shape=(h_depth,))
        self.bias_f = self.add_variable("bias_f",shape=(h_depth,))
        # cell
        self.W_xc = self.add_variable("w_xc",shape=(input_depth, h_depth))
        self.W_hc = self.add_variable("w_hc",shape=(h_depth, h_depth))
        self.bias_c = self.add_variable("bias_c",shape=(h_depth,))
        # output gate
        self.W_xo = self.add_variable("w_xo",shape=(input_depth, h_depth))
        self.W_ho = self.add_variable("w_ho",shape=(h_depth, h_depth))
        self.w_co = self.add_variable("w_co",shape=(h_depth,))
        self.bias_o= self.add_variable("bias_o",shape=(h_depth,))

        self.built = True

    def call(self, inputs, state):
        # time = tf.slice(inputs, [0, tf.shape(inputs)[1] - 1], [-1, 1], name="rnn_time")
        inputs = tf.slice(inputs, [0, 0], [-1, tf.shape(inputs)[1] - 1], name="rnn_input")
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        # cell clipping to avoid explostion
        c = clip_ops.clip_by_value(c, -1.0*_clipped_value, 1.0*_clipped_value)

        input_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xi),
                        math_ops.mat_mul(h,self.W_hi)),
                    math_ops.multiply(c,self.w_ci)),
                self.bias_i))

        forget_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xf),
                        math_ops.mat_mul(h,self.W_hf)),
                    math_ops.multiply(c,self.w_cf)),
                self.bias_f))
        new_c = math_ops.add(
            math_ops.multiply(forget_gate,c),
            math_ops.multiply(
                input_gate,math_ops.tanh(
                    math_ops.add(math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xc),
                        math_ops.mat_mul(h,self.W_hc)),
                        self.bias_c))))
        output_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xo),
                        math_ops.mat_mul(h,self.W_ho)),
                    math_ops.multiply(new_c,self.w_co)),
                self.bias_o))
        new_h = math_ops.multiply(output_gate,math_ops.tanh(new_c))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


class memory_cell(nlstm):
    def build(self, inputs_shape):
        print("memory cell")
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self.built = True

    def call(self, inputs, state):
        return inputs, state

def create_optimizer(learning_rate,name):
    if str(name).__contains__("adam"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif str(name).__contains__("adagrad"):
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif str(name).__contains__("sgd"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif str(name).__contains__("rms"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif str(name).__contains__("moment"):
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.1)
    return optimizer

class basic_model(object):
    GRAPHS = {}
    SESS = {}
    SAVER = {}

    @classmethod
    def create_model(cls, config, variable_scope = "target", trainable = True, graph_name="DEFAULT"):
        log.info("CREATE MODEL", config.model, "GRAPH", graph_name, "VARIABLE SCOPE", variable_scope)
        if not graph_name in cls.GRAPHS:
            log.info("Adding a new tensorflow graph:",graph_name)
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.Session(config = tf.ConfigProto(gpu_options=config.GPU_OPTION))
                cls.SAVER[graph_name] = tf.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {"graph": cls.GRAPHS[graph_name],
               "sess": cls.SESS[graph_name],
               "saver": cls.SAVER[graph_name],
               "model": model}

    def _update_placehoders(self):
        self.placeholders = {"none":{}}
        raise NotImplemented

    def _get_feed_dict(self,task,data_dicts):
        place_holders = self.placeholders[task]
        res = {}
        for key, value in place_holders.items():
            res[value] = data_dicts[key]
        return res

    def __init__(self, config, variable_scope = "target", trainable = True):
        print(self.__class__)
        self.config = config
        self.variable_scope = variable_scope
        self.trainable = trainable
        self.placeholders = {}
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.variable_scope):
            self._create_placeholders()
            self._create_global_step()
            self._update_placehoders()
            self._create_inference()
            if self.trainable:
                self._create_optimizer()
            self._create_intializer()

    def _create_global_step(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_intializer(self):
        with tf.name_scope("initlializer"):
            self.init = tf.global_variables_initializer()

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_inference(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def build_cell(self,rnn_type,initializer,hidden,input_data,initial_state):
        if rnn_type == "nlstm":
            cell = tf.contrib.rnn.MultiRNNCell([nlstm(hidden)]*self.config.RNN_LAYER)
            return tf.nn.dynamic_rnn(cell, input_data,
                                     initial_state=(tf.nn.rnn_cell.LSTMStateTuple(c=tf.zeros_like(initial_state[0]),
                                                                                  h=initial_state[0]),),
                                     dtype=tf.float32,
                                     time_major=True)
        elif rnn_type == "mem":
            cell = tf.contrib.rnn.MultiRNNCell([memory_cell(hidden)]*self.config.RNN_LAYER)
            return tf.nn.dynamic_rnn(cell, input_data,
                                     initial_state=(tf.nn.rnn_cell.LSTMStateTuple(c=tf.zeros_like(initial_state[0]),
                                                                                  h=initial_state[0]),),
                                     dtype=tf.float32,
                                     time_major=True)
