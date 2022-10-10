# -*- coding: utf-8 -*-
"""
This is the class to store the data information after preprocess
@author: cheng
"""

import glob
import os

class datainfo():
    
    def __init__(self, aug_num):        
        self.test_data = ['05_Trajectories',
                          '06_Trajectories',
                          '14_Trajectories', 
                          '15_Trajectories',
                          '16_Trajectories',
                          '17_Trajectories',
                          '26_Trajectories', 
                          '27_Trajectories',
                          '28_Trajectories',
                          '29_Trajectories',
                          '32_Trajectories']
        
        test_data_agu = []
        for test_dataname in self.test_data:
            for i in range(aug_num):
                if i == 0:
                    test_data_agu.append(test_dataname)
                else:
                    test_data_agu.append(test_dataname+"_%.0f"%(i))
        
        all_traindata_dirs = sorted(glob.glob(os.path.join("../processed_data/train", "*.npz")))
        train_data = []
        for train_dir in all_traindata_dirs:
            # train_dataname = train_dir.split('\\')[-1].split('.')[0]
            train_dataname = os.path.splitext(os.path.basename(train_dir))[0]
            if train_dataname not in train_data:
                # Remove the test_data from the train_data, including the augmented ones
                if train_dataname not in test_data_agu and train_dataname!='train_merged':               
                    train_data.append(train_dataname)
                               
        self.train_data = train_data
        self.train_merged = ['train_merged']        
        

        
        
        
        
    
        
        
                
            
                
            
                
            
            