#!/usr/bin/python
# encoding: utf-8



import numpy as np
import ipdb
import pandas as pd

def read_csv2data_frame(data_path = "",sep="\t",head=None,cols=[]):
    res = pd.read_csv(data_path,sep=sep,header=head,names=cols)
    return res