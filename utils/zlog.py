#!/usr/bin/python
# encoding: utf-8

from datetime import datetime

def get_now_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def generating_log(*info):
    temp = get_now_time() + " " + info[0]
    for item in info[1:]:
        temp += "\t" + str(item)
    temp += '\n'
    with open(log.log_path, 'a') as f:
        f.writelines(temp)
    print(temp.strip('\n'))

class log():
    log_path = "./logs/"+str(get_now_time())+".log"
    @classmethod
    def set_log_path(cls,path):
        cls.log_path = path

    @classmethod
    def redirect_log_path(cls,path):
        cls.log_path = path

    @classmethod
    def structure_info(cls,title="",info = []):
        temp = "#"*25+" "+str(title)+" "+"#"*25
        generating_log(temp)
        for item in info:
            generating_log(*item)
        generating_log(len(temp)*"#")

    @classmethod
    def info(cls,*info):
        generating_log(*info)





