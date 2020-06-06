#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
import scipy.stats as stats
from utils.statistical_utils import sigmoid,softmax
from progressbar import ProgressBar
import copy as cp


def load_matrix(path):
    with open(path,"r") as f:
        res = []
        for line in f.readlines():
           res.append(eval(line.strip("\n")))
    return np.asmatrix(res)

class ground_truth(object):
    all_score={}
    def __init__(self,config):
        self.config = config
        self.load_user_item_factor()

    def load_user_item_factor(self):
        self.u_factor = np.asarray(load_matrix(self.config.u_file))
        self.i_factor = np.asarray(load_matrix(self.config.i_file))
        self.entropy_matrix = np.asarray(load_matrix(self.config.entropy_file))

    def get_score(self,u,i):
        return 1/(1+ np.exp(-np.sum(np.multiply(self.u_factor[u],self.i_factor[i]))))

    def reset(self):
        self.u = np.random.choice(range(1,self.config.user_number),(1,))[0]
        return self.u

    def old_step(self,i):
        u = self.u
        score = self.get_score(u,i)
        if score >0.6:
            click = 1.0
            t_probability = 0.95
        else:
            click = 0.0
            t_probability = 0.7
        terminal = np.random.choice([0,1],p=[t_probability,1-t_probability])
        if terminal:
            self.reset()
        return u,click,terminal

    def random_action(self):
        return np.random.choice(range(self.config.item_number),(1,))[0]

    def sampling(self):
        try:
            self.all_score[self.u]
        except:
            self.all_score[self.u] = softmax(np.asarray([self.get_score(self.u, i) for i in range(self.config.item_number)]),0.1)
        action = np.random.choice(range(self.config.item_number),p=self.all_score[self.u])
        return action

    def get_probability(self,u,i):
        try:
            return self.all_score[u][i] * 0.5 + 0.5 / self.config.item_number
        except:
            self.all_score[u] = softmax(
                np.asarray([self.get_score(u, i) for i in range(self.config.item_number)]), 0.01)
            return self.all_score[u][i]*0.5+0.5/self.config.item_number

class diversity_environments(object):
    all_score_pos = {}
    all_score_neg = {}

    def __init__(self,config):
        self.config = config
        self.load_user_item_factor()

    def load_user_item_factor(self):
        self.u_factor = np.asarray(load_matrix(self.config.u_file))
        self.i_factor = np.asarray(load_matrix(self.config.i_file))
        self.entropy_matrix = np.asarray(load_matrix(self.config.entropy_file))

    def reset(self):
        self.u = np.random.choice(range(1, self.config.user_number), (1,))[0]
        self.trajectory = []
        self.try_good = np.random.choice([0,1,2])
        return self.u

    def get_entropy(self):
        if len(self.trajectory)<=1:
            return 1,0.0
        else:
            temp = []
            for i in self.trajectory[:-1]:
                temp.append(self.entropy_matrix[self.trajectory[-1],i])
            return 30* sigmoid(np.mean(temp)-self.config.threshold,1/30),np.mean(temp)

    def sampling(self):
        if np.random.uniform(0,1.0,(1,))[0]>0.5:
            self.try_good = 0
        else:self.try_good = 1
        if not self.u in self.all_score_neg:
            self.all_score_neg[self.u] = softmax(np.zeros((self.config.item_number,)),50.0)
            self.all_score_pos[self.u] = np.asarray([np.sum(np.multiply(self.u_factor[self.u],self.i_factor[i])) for i in range(self.config.item_number)])
        if self.try_good==0:
            temp = cp.deepcopy(self.all_score_pos[self.u])
            for item in self.trajectory: temp[item] = -10
            action = np.argmax(temp,0)
        elif self.try_good==1:
            action = np.random.choice(range(self.config.item_number), p=self.all_score_neg[self.u])
        return action

    def step(self,i):
        u = self.u
        score = self.get_score(u,i)
        self.trajectory.append(i)
        diversity,div = self.get_entropy()
        if score>0.5:click = 1.0
        else:click=0.0
        if click:
            t_probability = 0.9*diversity
        else:
            t_probability = 0.5*diversity
        terminal = np.random.choice([0,1],p=[t_probability,1-t_probability])
        if len(self.trajectory)>=10: terminal=True
        return u,click,terminal,div


