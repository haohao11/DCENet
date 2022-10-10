# -*- coding: utf-8 -*-
"""

@author: cheng
"""
import os, errno

def mak_dir():
    # Make all the folders to save the intermediate results
    model_dir = "../models"
    processed_train = "../processed_data/train"
    # Save the model's prediction
    prediction = "../prediction"

    
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    try:
        os.makedirs(processed_train)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
    try:
        os.makedirs(prediction)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    
    
    
    