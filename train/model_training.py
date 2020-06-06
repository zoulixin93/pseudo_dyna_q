#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
from function_approximation import reward_network,rl_network,q_network
from environment import diversity_environments
from utils.zlog import log
from utils.data_structure import type_memory
from utils.time_analyse import timeit

def sigmoid(x,s):
    return s/(1+np.exp(-x/s))

def softmax(x, temperature = 0.1):
    return np.exp(x/temperature)/np.sum(np.exp(x/temperature))

class base(object):
    def __init__(self,config):
        self.config = config
        self.user_item_visit_count = np.zeros((self.config.user_number,self.config.item_number))
        self.init_model()
        pass

    def init_model(self):
        raise NotImplementedError

    def update_model(self,data_set=[]):
        pass

    def get_all_prediction(self,state,trajectory):
        raise NotImplementedError
        pass

    def get_action(self,state,trajectory):
        action_score = self.get_all_prediction(state,trajectory)
        return np.argmax(action_score, 0)

    def get_epsilon_greedy(self,state,trajectory,epsilon=0.05):
        if np.random.uniform(0,1,(1,))>(1-epsilon):
            action = np.random.choice(range(self.config.item_number))
        else:
            action = self.get_action(state,trajectory)
        return action

    def tompson_sampling(self,state,trajectory,temp=0.001):
        action_score = (self.get_all_prediction(state, trajectory))/temp
        action_score = softmax(action_score)
        try:
            action = np.random.choice(range(self.config.item_number),p=action_score)
        except:
            action = np.random.choice(range(self.config.item_number))
        return action

    def epsilon_greedy(self,state,trajectory,epsilon):
        action_score = self.get_all_prediction(state, trajectory)
        if np.random.uniform(0,1.0,(1,))[0]<=epsilon:
            action = np.random.choice(range(self.config.item_number))
        else:
            action = np.argmax(action_score)
        return action

    def evaluate_or_not(self):
        return True

class agent_dqn(base):
    def init_model(self):
        self.agent = q_network.create_model(config=self.config)
        self.memory = []

    def get_next_q_value_tuple(self,in_data):
        res = []
        uid = [item[0] for item in in_data if item[-1]!=-1]
        trajectory = [item[-1] for item in in_data if item[-1]!=-1]
        # ipdb.set_trace()
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[j[1] for j in item] for item in trajectory],
                                                                     [[j[2] for j in item] for item in trajectory])
        data = {"uid": uid,"trajectory": traj,"target_index": target_index,"feedback": feedbacks}
        prediction = np.max(self.agent["model"].predict_all(self.agent["sess"], data),1)
        index = 0
        for item in in_data:
            if item[-1]==-1:
                next_q = 0
            else:
                next_q = prediction[index]
                index+=1
            res.append((item[0],item[1],item[2],item[3]+self.config.gamma*next_q))
        return res,prediction

    def update_model(self,data_set=[]):
        if np.mean([len(item) for item in data_set])==1.0:
            return 0.0
        temp_data = []
        for trajectory in data_set:
            for i,sar in enumerate(trajectory):
                state,action,reward = sar
                if i==len(trajectory)-1: temp_data.append((state,action,trajectory[:i],reward,-1))
                else:temp_data.append((state,action,trajectory[:i],reward,trajectory[:i+1]))
        temp_data,next_max_q = self.get_next_q_value_tuple(temp_data)
        self.memory.extend(temp_data)
        if len(self.memory)>=self.config.buffer_size: self.memory = self.memory[-self.config.buffer_size:]
        batch = [self.memory[item] for item in np.random.choice(len(self.memory), (self.config.batch_size,))]
        uid = [i[0] for i in batch]
        iid = [i[1] for i in batch]
        label = [i[3] for i in batch]
        ts = [i[2] for i in batch]
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[ii[1] for ii in item] for item in ts],
                                                                     [[ii[2] for ii in item] for item in ts])
        data = {"uid":uid,"iid":iid,"label":label,"trajectory":traj,"feedback":feedbacks,"target_index":target_index}
        loss = self.agent["model"].optimize_model(self.agent["sess"], data)
        log.info("average max_next_q value",np.mean(next_max_q))
        return loss

    def get_all_prediction(self,state=0,trajectory=[]):
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[item[1] for item in trajectory]],
                                                                     [[item[2] for item in trajectory]])
        data = {"uid": [state],"trajectory": traj,"target_index": target_index,"feedback": feedbacks}
        prediction = self.agent["model"].predict_all(self.agent["sess"],data)[0]
        return prediction

    def get_weights(self,state=[],trajectory=[],feedback = [],actions=[]):
        traj, feedbacks, target_index = self.convert_item_seq2matrix(trajectory,feedback)
        data = {"uid": state,"iid":actions,"trajectory": traj,"target_index": target_index,"feedback": feedbacks}
        prediction = self.agent["model"].predict_f(self.agent["sess"],data)/30
        return prediction

    def convert_item_seq2matrix(self,item_seq,feeds):
        max_length = max([len(item) for item in item_seq])
        matrix = np.zeros((max_length,len(item_seq)))
        feedback = np.zeros((max_length,len(item_seq)))
        for x,xx in enumerate(item_seq):
            if len(xx)>0:
                for y,yy in enumerate(xx):
                    matrix[y,x] = yy
                    feedback[y,x] = feeds[x][y]
            else:continue
        target_index = list(zip([len(i) - 1 for i in item_seq], range(len(item_seq))))
        if sum([len(item) for item in item_seq])== 0:
            matrix = [[0]]*len(item_seq)
            feedback = [[0]]*len(item_seq)
        return matrix,feedback,target_index

class model_training(agent_dqn):
    global_update_time = 0
    epoch = 0
    def init_model(self):
        self.memory = type_memory(4,int(self.config.buffer_size/4))
        self.simulator = reward_network.create_model(config=self.config,
                                                     variable_scope="simulator",
                                                     trainable=True,
                                                     graph_name="simulator")
        self.rec_agent = agent_dqn(self.config)
        self.env = diversity_environments(self.config)
        self.update_env()

    def get_next_q_value_tuple(self,in_data):
        res = []
        uid = [item[0] for item in in_data if item[-1]!=-1]
        trajectory = [item[-1] for item in in_data if item[-1]!=-1]
        # ipdb.set_trace()
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[j[1] for j in item] for item in trajectory],
                                                                     [[j[2] for j in item] for item in trajectory])
        data = {"uid": uid,"trajectory": traj,"target_index": target_index,"feedback": feedbacks}
        prediction = np.max(self.agent["model"].predict_all(self.agent["sess"], data),1)
        index = 0
        for item in in_data:
            if item[-1]==-1:
                next_q = 0
            else:
                next_q = prediction[index]
                index+=1
            res.append((item[0],item[1],item[2],item[3]+self.config.gamma*next_q))
        return res,prediction

    def update_env(self):
        self.env.memory_capacitycreate_simulator_latent_factor.py(self.simulator)

    def update_model(self, data_set=[]):
        self.global_update_time += 1
        if self.global_update_time <= self.config.evaluation_num:
            loss = self.update_simulator(data_set)
            self.update_env()
        else:
            if self.global_update_time%10==0:
                self.update_rec_agent()
            loss = self.update_simulator(data_set)
            self.update_env()
        return loss

    def update_simulator(self, data_set=[]):
        for trajectory in data_set:
            for i, sar in enumerate(trajectory):
                state, action, reward = sar
                if reward>0:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0),0)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0), 1)
                else:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0), 2)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0), 3)
        batch = self.memory.sample_batch(self.config.batch_size)
        uid = [i[0] for i in batch]
        iid = [i[1] for i in batch]
        label = [i[3] for i in batch]
        ts = [i[2] for i in batch]
        terminate = [i[4] for i in batch]
        log.info("training ratio",np.mean(terminate),"reward",np.mean(label))
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[ii[1] for ii in item] for item in ts],
                                                                     [[ii[2] for ii in item] for item in ts])
        weight = [1.0]*len(label)
        data = {"uid": uid,
                "iid": iid,
                "label": label,
                "trajectory": traj,
                "feedback": feedbacks,
                "target_index": target_index,
                "terminate": terminate,
                "weight":weight}
        loss = self.simulator["model"].optimize_model(self.simulator["sess"], data)
        log.info("loss for simulator",loss)
        return loss

    def update_rec_agent(self):
        for i in range(1):
            self.collecting_training_data(self.epoch)
            loss = self.rec_agent.update_model(self.data_set)
            log.info("rec agent", i, 'loss', loss)
            self.epoch+=1

    def evaluate_or_not(self):
        if self.epoch%200==0:
            return True

    @timeit
    def collecting_training_data(self,epoch):
        self.data_set = []
        log.info("temperature is ",epoch,max([self.config.temperature/(epoch+1e-10),self.config.mini_temperature]))
        for _ in range(64):
            rewards = 0
            trajectory = []
            cur_state = self.env.reset()
            terminal = False
            while not terminal:
                action = self.rec_agent.tompson_sampling(cur_state,trajectory,max([self.config.temperature/(epoch+0.1e10),self.config.mini_temperature]))
                next_state, reward, terminal = self.env.step(action)
                trajectory.append((cur_state, action, reward))
                cur_state = next_state
                rewards += reward
            self.data_set.append(trajectory)
        log.info("fake environment",self.epoch,
                 "clicks",np.mean([np.sum([i[2] for i in item]) for item in self.data_set]),
                 "depth",np.mean([len(item) for item in self.data_set]))

    def get_all_prediction(self,state=0,trajectory=[]):
        return self.rec_agent.get_all_prediction(state,trajectory)

    def get_weights(self,state=[],trajectory=[],feedback = [],actions=[]):
        traj, feedbacks, target_index = self.convert_item_seq2matrix(trajectory,feedback)
        data = {"uid": state,"iid":actions,"trajectory": traj,"target_index": target_index,"feedback": feedbacks}
        prediction = self.agent["model"].predict_f(self.agent["sess"],data)/30
        return prediction

    def convert_item_seq2matrix(self,item_seq,feeds):
        max_length = max([len(item) for item in item_seq])
        matrix = np.zeros((max_length,len(item_seq)))
        feedback = np.zeros((max_length,len(item_seq)))
        for x,xx in enumerate(item_seq):
            if len(xx)>0:
                for y,yy in enumerate(xx):
                    matrix[y,x] = yy
                    feedback[y,x] = feeds[x][y]
            else:continue
        target_index = list(zip([len(i) - 1 for i in item_seq], range(len(item_seq))))
        if sum([len(item) for item in item_seq])== 0:
            matrix = [[0]]*len(item_seq)
            feedback = [[0]]*len(item_seq)
        return matrix,feedback,target_index

class model_training_with_weight(model_training):
    def get_action_probability(self,data_set = []):
        s,a,t,r = [],[],[],[]
        for trajectory in data_set:
            seq = []
            rewards = []
            for i, sar in enumerate(trajectory):
                state, action, reward = sar
                s.append(state)
                a.append(action)
                t.append(seq)
                r.append(rewards)
                seq.append(action)
                rewards.append(reward)
        return self.rec_agent.get_weights(s,t,r,a)

    def update_simulator(self, data_set=[]):
        action_probability = self.get_action_probability(data_set)
        index = 0
        for trajectory in data_set:
            probability = 1.0
            for i, sar in enumerate(trajectory):
                state, action, reward = sar
                probability *= action_probability[index]/self.env.get_probability(state,action)
                pp_a = min([probability,self.config.maximum_weight])
                if reward>0:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0,pp_a),0)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0, pp_a), 1)
                else:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0, pp_a), 2)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0, pp_a), 3)
                index+=1
        batch = self.memory.sample_batch(self.config.batch_size)
        uid = [i[0] for i in batch]
        iid = [i[1] for i in batch]
        label = [i[3] for i in batch]
        ts = [i[2] for i in batch]
        terminate = [i[4] for i in batch]
        weight = [i[5] for i in batch]
        log.info("training ratio",np.mean(terminate),"reward",np.mean(label),"weight",np.mean(weight))
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[ii[1] for ii in item] for item in ts],
                                                                     [[ii[2] for ii in item] for item in ts])
        data = {"uid": uid,
                "iid": iid,
                "label": label,
                "trajectory": traj,
                "feedback": feedbacks,
                "target_index": target_index,
                "terminate": terminate,
                "weight":weight}
        loss = self.simulator["model"].optimize_model(self.simulator["sess"], data)
        p_c, p_t = self.simulator["model"].predict(self.simulator["sess"], data)
        self.env.set_click_terminate_threshold(np.mean(p_c),np.mean(p_t))
        return loss


class model_training_with_weighted_regularizer(model_training_with_weight):
    negative_training_sample = []
    def update_model(self, data_set=[]):
        self.global_update_time += 1
        if self.global_update_time <= self.config.evaluation_num:
            loss = self.update_simulator(data_set)
            self.update_env()
        else:
            if self.global_update_time%10==0:
                self.collecting_training_data(self.epoch)
                self.update_rec_agent()
                loss_neg = self.update_simulator_negative()
                log.info("simulator negative loss",loss_neg)
            loss = self.update_simulator(data_set)
            log.info("simulator positive loss",loss)
            self.update_env()
        return loss

    @timeit
    def update_simulator(self, data_set=[]):
        action_probability = self.get_action_probability(data_set)*0.01
        index = 00
        for trajectory in data_set:
            probability = 1.0
            for i, sar in enumerate(trajectory):
                state, action, reward = sar
                probability *= action_probability[index]/self.env.get_probability(state,action)
                pp_a = min([probability,self.config.maximum_weight])
                if reward>0:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0,pp_a),0)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0,pp_a),1)
                else:
                    if i == len(trajectory) - 1:
                        self.memory.put((state, action, trajectory[:i], reward, 1.0,pp_a),2)
                    else:
                        self.memory.put((state, action, trajectory[:i], reward, 0.0,pp_a),3)
                index+=1
        batch = self.memory.sample_batch(self.config.batch_size)
        uid = [i[0] for i in batch]
        iid = [i[1] for i in batch]
        label = [i[3] for i in batch]
        ts = [i[2] for i in batch]
        terminate = [i[4] for i in batch]
        weight = [i[5] for i in batch]
        log.info("training ratio",np.mean(terminate),"reward",np.mean(label),"weight",np.mean(weight))
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[ii[1] for ii in item] for item in ts],
                                                                     [[ii[2] for ii in item] for item in ts])
        data = {"uid": uid,
                "iid": iid,
                "label": label,
                "trajectory": traj,
                "feedback": feedbacks,
                "target_index": target_index,
                "terminate": terminate,
                "weight":weight}
        loss = self.simulator["model"].optimize_model(self.simulator["sess"], data)
        p_c, p_t = self.simulator["model"].predict(self.simulator["sess"], data)
        self.env.set_click_terminate_threshold(np.mean(p_c),np.mean(p_t))
        log.info("loss for simulator",loss)
        return loss

    @timeit
    def update_simulator_negative(self):
        for trajectory in self.data_set:
            for i, sar in enumerate(trajectory):
                state, action, reward = sar
                if i == len(trajectory) - 1:self.negative_training_sample.append((state,action,trajectory[:i],reward,1.0))
                else: self.negative_training_sample.append((state,action,trajectory[:i],reward,0.0))
        if len(self.negative_training_sample) >= 50 * self.config.batch_size: self.negative_training_sample = self.negative_training_sample[
                                                                                  -50 * self.config.batch_size:]
        batch = [self.negative_training_sample[item] for item in np.random.choice(len(self.negative_training_sample), (4*self.config.batch_size,))]
        uid = [i[0] for i in batch]
        iid = [i[1] for i in batch]
        label = [i[3] for i in batch]
        ts = [i[2] for i in batch]
        terminate = [i[4] for i in batch]
        weight = [0.0 for i in batch]
        for i,item in enumerate(ts):
            if len(item) ==0: ts[i]= [(0,0,0)]
        traj, feedbacks, target_index = self.convert_item_seq2matrix([[ii[1] for ii in item] for item in ts],
                                                                     [[ii[2] for ii in item] for item in ts])
        data = {"uid": uid,
                "iid": iid,
                "label": label,
                "trajectory": traj,
                "feedback": feedbacks,
                "target_index": target_index,
                "terminate": terminate,
                "weight":weight}
        loss = self.simulator["model"].optimize_model_neg(self.simulator["sess"], data)
        return loss
