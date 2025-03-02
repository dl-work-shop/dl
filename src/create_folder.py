import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
from get_data import read_params,get_data

############################# Creating Folder - STAR #############################

def create_folder(config, image=None):
    config = get_data(config)
    dir = config['load_data']['preprocess_data']
    cla = config['load_data']['num_classes']

    if os.path.exists(dir+'/'+'train'+'/'+'class_0') and os.path.exists(dir+'/'+'test'+'/'+'class_0'):
        print('Folders already exists')
        print("I am skipping it............")
    else:
        os.mkdir(dir+'/'+'train')
        os.mkdir(dir+'/'+'test')
        for i in range(cla):
            os.mkdir(dir+'/'+'train'+'/'+'class_'+str(i))
            os.mkdir(dir+'/'+'test'+'/'+'class_'+str(i))

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args = args.parse_args()
    create_folder(config=passed_args.config)