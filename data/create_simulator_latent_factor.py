#!/usr/bin/python
# encoding: utf-8
import numpy as np
import ipdb
import tensorflow as tf
import os
from progressbar import ProgressBar
root_path = "./Tmall/"
data_path ="tmall"
u_path = "u_factor"
i_path = "i_factor"
e_path = "entropy"
u_num = 100000
i_num = 420000
hidden_state = 30
learning_rate = 0.1
maximum_epoch = 3000

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
def save_numpy_array(data_array,path):
    with open(path,"w") as f:
        for i in range(data_array.shape[0]):
            f.writelines(str(list(data_array[i,:]))+"\n")

p_uid = tf.placeholder(tf.int32,(None,), name="user_id")
p_pos_item = tf.placeholder(tf.int32,(None,), name="postive_sample")
p_neg_item = tf.placeholder(tf.int32,(None,), name="negative_sample")
p_label = tf.placeholder(tf.float64,(None,), name="training_label")
u_factor = tf.Variable(np.random.uniform(-0.1,0.1,(u_num,hidden_state)),name='u_factor')
i_factor = tf.Variable(np.random.uniform(-0.1,0.1,(i_num,hidden_state)),name='i_factor')
i_norm_factor = tf.nn.softmax(i_factor,1)
u_embedding = tf.nn.embedding_lookup(u_factor,p_uid)
pos_feature = tf.nn.embedding_lookup(i_factor,p_pos_item)
neg_feature = tf.nn.embedding_lookup(i_factor,p_neg_item)
p_score = tf.reshape(tf.reduce_sum(tf.multiply(u_embedding, pos_feature), axis=1), (-1,))
n_score = tf.reshape(tf.reduce_sum(tf.multiply(u_embedding, neg_feature), axis=1), (-1,))
loss = tf.reduce_sum(-tf.log(tf.sigmoid(p_score))-tf.log(1.0-tf.sigmoid(n_score)))
optimizer = tf.train.GradientDescentOptimizer(0.1)
gvs = optimizer.compute_gradients(loss)
optimizer = optimizer.apply_gradients(gvs)
sess = tf.Session(config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
init = tf.global_variables_initializer()
sess.run(init)
### load training data
uid2pos = {}
uid2neg = {}
with open(os.path.join(root_path,data_path),"r") as f:
    for line in f.readlines():
        temp = [i for i in line.strip("\n").split("\t")]
        uiid = int(temp[0])
        iiid = [i.split(";")[0] for i in temp[1:]]
        uid2pos[uiid] = np.asarray(iiid)
        uid2neg[uiid] = np.asarray(list(set(range(i_num)).difference(set(list(iiid)))))
print("finish")
memory = []
memory_capacity = 1000
batch = 300
update_num = 1
while True:
    keys = list(uid2pos.keys())
    np.random.shuffle(keys)
    for key in keys:
        pos = np.random.choice(uid2pos[key])
        neg = np.random.choice(uid2neg[key])
        memory.append((key,pos,neg,1.0))
        if len(memory)>=memory_capacity and len(memory)%batch==0:
            print("running")
            update_num+=1
            if len(memory)>memory_capacity: memory = memory[batch:]
            batch_memory = [memory[i] for i in np.random.choice(range(len(memory)),size = (batch,))]
            t_uids,t_pos,t_neg,t_label = [list(item) for item in zip(*batch_memory)]
            t_loss,_ = sess.run([loss,optimizer],
                                feed_dict={p_uid:t_uids,
                                           p_pos_item:t_pos,
                                           p_neg_item:t_neg,
                                           p_label:t_label})
            print("training loss",update_num,t_loss)
            if update_num%maximum_epoch==0:
                t_u_factor = sess.run(u_factor)
                t_i_factor = sess.run(i_factor)
                ii_norm_factor = sess.run(i_norm_factor)
                res = []
                tt_p = np.matmul(t_u_factor,np.transpose(t_i_factor))
                for i in range(i_num):
                    temp = list(tt_p[:,i])
                    res.append(len([item for item in temp if item>0.5]))
                print(res)
                print(np.sum(res))
                res = []
                for i in range(u_num):
                    temp = list(tt_p[i,:])
                    res.append(len([item for item in temp if item>0.5]))
                print(res)
                save_numpy_array(t_u_factor,os.path.join(root_path,u_path))
                save_numpy_array(t_i_factor,os.path.join(root_path,i_path))
                entropy_matrix = np.zeros((i_num, i_num))
                pbar = ProgressBar()
                for i in pbar(range(i_num)):
                    for j in range(i_num):
                        entropy_matrix[i, j] = np.sum(ii_norm_factor[i] * np.log(ii_norm_factor[i] / ii_norm_factor[j]))
                save_numpy_array(entropy_matrix, os.path.join(root_path, e_path))
                print(np.mean(entropy_matrix))
                break
    if update_num%maximum_epoch==0:
       break








