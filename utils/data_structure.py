#!/usr/bin/python
# encoding: utf-8



import numpy as np
import ipdb


class random_buffer(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
    def put(self,data):
        self.buffer+=data
        if len(self.buffer)>self.capacity:
            self.buffer = self.buffer[len(self.buffer)-self.capacity:]
    def get(self,size):
        index = np.random.choice(range(len(self.buffer)),size)
        return [self.buffer[i] for i in index]

    def get_buffer_size(self):
        return len(self.buffer)

    def get_all(self):
        return self.buffer

    def full(self):
        if len(self.buffer) >= self.capacity:
            return True
        else:
            return False


class type_memory(object):
    def __init__(self,kinds=4,max_buffer=100):
        self.memory = []
        self.max_buffer = max_buffer
        for i in range(kinds):
            self.memory.append([])

    def put(self,data,type_num):
        self.memory[type_num].append(data)
        self._clear_large(type_num)

    def clear(self):
        for i in range(len(self.memory)): self.memory[i] = []

    def _clear_large(self,type_num):
        if len(self.memory[type_num])>self.max_buffer:self.memory[type_num] = self.memory[type_num][-self.max_buffer:]

    def sample_batch(self,batch_size):
        batch = []
        for mem in self.memory:
            batch.extend([mem[item] for item in np.random.choice(len(mem), (batch_size,))])
        return batch






