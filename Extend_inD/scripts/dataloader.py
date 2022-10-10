# -*- coding: utf-8 -*-
"""
This is the load to load data based on occupancy maps

@author: cheng
"""

import numpy as np
import time
import os

from augmentation import rotation
from maps import Maps
from occupancy import dynamic_maps
import pathlib
    
def loaddata(dataset_list, args, datatype="train"):
    # Store the data across datasets
    if datatype=="train" or datatype=="test":
        offsets = np.empty((0, args.obs_seq+args.pred_seq-1, 8))
        traj_data = np.empty((0, args.obs_seq+args.pred_seq, 4))
        occupancy = np.empty((0, args.obs_seq+args.pred_seq-1, args.enviro_pdim[0], args.enviro_pdim[1], 3))
        
        for i, dataset in enumerate(dataset_list):
            # Only take the original data
            # ToDo, here needs to be test if augumentation will boost the performance
            if i%4 in (0, 1, 2, 3):
                data = np.load(pathlib.Path(__file__).parent / f"../processed_data/train/{dataset}.npz")
                _offsets, _traj_data, _occupancy = data["offsets"], data["traj_data"], data["occupancy"]
                
                print(dataset, "contains %.0f trajectories"%len(_offsets))    
                offsets = np.concatenate((offsets, _offsets[:, :args.obs_seq+args.pred_seq-1, :]), axis=0)
                traj_data = np.concatenate((traj_data, _traj_data[:, :args.obs_seq+args.pred_seq, :]), axis=0)
                occupancy = np.concatenate((occupancy, _occupancy[:, :args.obs_seq+args.pred_seq-1, :]), axis=0)
                                
                    
    elif datatype == "challenge":      
        offsets = np.empty((0, args.obs_seq-1, 8))
        traj_data = np.empty((0, args.obs_seq, 4))
        occupancy = np.empty((0, args.obs_seq-1, args.enviro_pdim[0], args.enviro_pdim[1], 3))        
        for dataset in dataset_list:
            data = np.load(pathlib.Path(__file__).parent / f"../processed_data/challenge/{dataset}.npz")
            _offsets, _traj_data, _occupancy = data["offsets"], data["traj_data"], data["occupancy"]
            offsets = np.concatenate((offsets, _offsets[:, :args.obs_seq+args.pred_seq-1, :]), axis=0)
            traj_data = np.concatenate((traj_data, _traj_data[:, :args.obs_seq+args.pred_seq, :]), axis=0)
            occupancy = np.concatenate((occupancy, _occupancy[:, :args.obs_seq+args.pred_seq-1, :]), axis=0)
                    
        
    elif datatype=="test":
        assert len(dataset_list)==1, print("Only one untouched dataset is left fot testing!")
         
    elif datatype=="challenge":
        assert len(dataset_list)==1, print("predict one by one")
        
        
    if datatype=="train":
        if not os.path.exists(pathlib.Path(__file__).parent / "../processed_data/train/train_merged.npz"):
            # Save the merged training data
            np.savez(pathlib.Path(__file__).parent / "../processed_data/train/train_merged.npz", 
                     offsets=offsets,
                     traj_data = traj_data,
                     occupancy = occupancy)
            
        
                
    return offsets, traj_data, occupancy
    


def preprocess_data(seq_length, size, dirname, path=None, data=None, aug_num=1, save=True):
    '''
    Parameters
    ----------
    seq_length : int
        This is the complete length of each trajectory offset and occupancy, 
        Note: one-step difference for the offset and occupancy and traj_data.
    size : [height, width, channels]
        The occupancy grid size and channels: 
            orientation, speed and position for the neighbors in the vicinity
    dirname : string
        "train" or "challenge"
    path : string, optional
        only for extract offsets, traj_data, and occupancy from the original data files
    data : numpy, optional
        it is the predicted complete trajectories after the first prediction,
        it is used to calculate the occupancy in the predicted time.
    aug_num : int, optional
        the number for augmenting the data by rotation.
    save : boolen, optional
        Only save the processed training data. The default is True.

    Returns
    -------
    offsets : numpy array
        [frameId, userId, x, y, delta_x, delta_y, theata, velocity].
    traj_data : numpy array
        [frameId, userId, x, y]
        Note: this is one-step longer 
    occupancy : numpy array
        [height, width, channels].
    '''    
    start = time.time()
    if np.all(data)==None:
        data = np.genfromtxt(path, delimiter='')
        print('path', path)
        # challenge dataset have nan for prediction time steps        
        data = data[~np.isnan(data).any(axis=1)]              
        # dataname = path.split('\\')[-1].split('.')[0]
        dataname = os.path.splitext(os.path.basename(path))[0]
        print("process data %s ..."%dataname)
        
    
    for r in range(aug_num):        
        # Agument the data by orientating if the agumentation number if more than one
        if r > 0:
            data[:, 2:4] = rotation(data[:, 2:4], r/aug_num)            
    
        # Get the environment maps
        maps = Maps(data)    
        traj_map = maps.trajectory_map()
        orient_map, speed_map = maps.motion_map(max_speed=10)   
        map_info = [traj_map, orient_map, speed_map]
        enviro_maps = concat_maps(map_info)
        # print("enviro_maps shape", enviro_maps.shape)
        
        print('maps.offsets', maps.offsets.shape)
            
        offsets = np.reshape(maps.offsets, (-1,seq_length,8))
        # print("offsets shape", offsets.shape)
        traj_data = np.reshape(maps.sorted_data, (-1, seq_length+1, 4)) 
        # print("traj_data shape", traj_data.shape)
        occupancy = dynamic_maps(offsets, maps.sorted_data, size)
        # print("occupancy shape", occupancy.shape)   
          
        if save: 
            if r == 0:
                # Save the original one
                np.savez(pathlib.Path(__file__).parent / f"../processed_data/{dirname}/{dataname}", 
                         offsets=offsets,
                         traj_data = traj_data,
                         occupancy = occupancy)
                end = time.time() 
                
            else:
                # Save the rotated one(s)
                np.savez(pathlib.Path(__file__).parent / f"../processed_data/{dirname}/{dataname}_{r}", 
                         offsets=offsets,
                         traj_data = traj_data,
                         occupancy = occupancy)
                end = time.time() 
            print("It takes ", round(end-start, 2), "seconds!\n")    
            
        else:
            return offsets, traj_data, occupancy


def concat_maps(map_info):
    # save the map information into different channels
    enviro_maps = np.empty((map_info[0].shape[0], map_info[0].shape[1], len(map_info)))
    for i, map in enumerate(map_info):
        enviro_maps[:, :, i] = map
    return enviro_maps

    

