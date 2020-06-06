#!/usr/bin/python
# encoding: utf-8

import numpy as np
import ipdb
from environment import ground_truth,diversity_environments
from utils.zlog import log
from progressbar import ProgressBar
from utils.time_analyse import timeit
import time
import os
import random

class training(object):
    def __init__(self,config):
        self.config = config
        self.init_training()

    def init_training(self):
        log.info("load environment")
        self.training_agent = self.config.training_agent(self.config)
        self.env = diversity_environments(self.config)
        self.data_set = []
        self.collecting_training_data()


    def collecting_training_data(self):
        if os.path.isfile(self.config.save_path):
            with open(self.config.save_path,"r") as f:
                for line in f.readlines():
                    self.data_set.append(eval(line.strip("\n")))
        else:
            pbar = ProgressBar()
            diversitys = []
            for _ in pbar(range(self.config.trajectory_number)):
                trajectory = []
                cur_state = self.env.reset()
                terminal = False
                while not terminal:
                    action = self.env.sampling()
                    next_state,reward,terminal,div = self.env.step(action)
                    trajectory.append((cur_state,action,reward))
                    diversitys.append(div)
                    cur_state = next_state
                self.data_set.append(trajectory)
            log.info("finish collecting training data",
                     np.mean([len(item) for item in self.data_set]),
                     "average click",
                     np.mean([np.sum([i[2] for i in item]) for item in self.data_set]),
                     "diversity",np.mean(diversitys))
            with open(self.config.save_path,"w") as f:
                for item in self.data_set:
                    f.writelines(str(item)+"\n")

    def run(self):
        for i in range(self.config.epoch):
            if i%200==0 and i>=self.config.evaluation_num and self.training_agent.evaluate_or_not():
                click,length,div = self.evaluate()
                log.info("epoch",i,"average click",click,"depth",length,"diversity",div)
            random.shuffle(self.data_set)
            batch = [self.data_set[item] for item in np.random.choice(len(self.data_set),(self.config.batch_size,))]
            loss = self.training_agent.update_model(batch)
            log.info("training epoch",i,'loss',loss)

    def evaluate(self):
        all_reward = []
        all_length = []
        diversitys = []
        for i in range(1,self.config.user_number):
            rewards = 0
            trajectory = []
            cur_state = i
            self.env.reset()
            self.env.u = cur_state
            terminal = False
            while not terminal:
                action = self.training_agent.get_action(cur_state,trajectory)
                next_state,reward ,terminal,div = self.env.step(action)
                trajectory.append((cur_state, action, reward))
                diversitys.append(div)
                cur_state = next_state
                rewards += reward
            all_reward.append(rewards)
            all_length.append(len(trajectory))
        print(np.mean(all_reward),np.mean(all_length),np.mean(diversitys))
        return np.mean(all_reward),np.mean(all_length),np.mean(diversitys)

class online_training(training):
    def collecting_training_data(self,epoch=1):
        self.data_set = []
        diversitys = []
        for _ in range(64):
            rewards = 0
            trajectory = []
            cur_state = self.env.reset()
            terminal = False
            while not terminal:
                action = self.training_agent.tompson_sampling(cur_state,trajectory,max([self.config.temperature/(epoch+1e-10),self.config.mini_temperature]))
                next_state,reward,terminal,div = self.env.step(action)
                diversitys.append(div)
                trajectory.append((cur_state, action, reward))
                cur_state = next_state
                rewards += reward
            self.data_set.append(trajectory)
        log.info("finish collecting training data",
                 np.mean([len(item) for item in self.data_set]),
                 "average click",
                 np.mean([np.sum([i[2] for i in item]) for item in self.data_set]),
                 "average depth",
                 np.mean([len(item) for item in self.data_set]),
                 "diversity",
                 np.mean(diversitys))


    def run(self):
        for i in range(self.config.epoch):
            self.collecting_training_data(i)
            loss = self.training_agent.update_model(self.data_set)
            log.info("training epoch",i,'loss',loss)
            if i%200==0:
                click,length,diversity = self.evaluate()
                log.info("epoch",i,"average click",click,"depth",length,"div",np.mean(diversity))










