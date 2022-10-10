# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:10:36 2020
This is the function to plot the trajectories from InD dataset
https://www.ind-dataset.com/
@author: cheng
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pathlib

np.set_printoptions(suppress=True)


def main():
    dowmsampling = True
    
    path = '/Users/angtoy/Documents/Datasets/inD-dataset-v1.0/data'
    tracks_dirs = sorted(glob.glob(os.path.join(path, "*_tracks.csv")))
    rMate_dirs = sorted(glob.glob(os.path.join(path, "*_recordingMeta.csv")))
    tMeta_dirs = sorted(glob.glob(os.path.join(path, "*_tracksMeta.csv")))
    bgimage_dirs = sorted(glob.glob(os.path.join(path, "*.png")))    
    
    for i, tracks_dir in enumerate(tracks_dirs):
        tracks = pd.read_csv(tracks_dir)
        recordingMeta = pd.read_csv(rMate_dirs[i])
        tracksMeta = pd.read_csv(tMeta_dirs[i])
        
        ori_framerate = recordingMeta['frameRate'].values[0]
        dowmsample_rate = ori_framerate / 2.5 
        
# =============================================================================
#         # Down sample the tracks from the original frame rate t0 2.5 fps (timestep=0.4s)
#         if dowmsampling:                   
#             tracks = tracks.loc[tracks['frame']%dowmsample_rate==0]
#         
#         # # Plot the trajectories on the background image
#         orthoPxToMeter = recordingMeta['orthoPxToMeter'].values[0]*12        
#         plot(tracks, bgimage_dirs[i], orthoPxToMeter, tracksMeta, i)
# =============================================================================

        seq_index = get_seq_index(tracksMeta, dowmsample_rate)
        traj = extract_data(tracks, seq_index, dowmsample_rate)
        np.savetxt(pathlib.Path(__file__).parent / '../trajectories_InD' / f"{i:02}_Trajectories.txt", traj)
        
        with open(pathlib.Path(__file__).parent / '../trajectories_InD' / f"{i:02}_Trajectories.txt","w+") as f:
            for t in traj:
                f.write(str(int(t[0]))+' '+
                             str(round(t[1], 0))+' '+
                             str(round(t[2], 5))+' '+
                             str(round(t[3], 5))+' '+'\n')
        f.close()
        
        # sys.exit()


def plot(data, background, orthoPxToMeter, tracksMeta, name):
    fig, ax = plt.subplots()
    img = plt.imread(background)
    ax.imshow(img)
    color_dic = {'pedestrian':'b',  'bicycle':'g', 'car':'r', 'truck_bus':'c'}

    count_length = []    

    trackIds = data.trackId.unique()
    for trackId in trackIds:
        user_info = tracksMeta.loc[tracksMeta['trackId'] == trackId].values        
        user_traj = data.loc[data['trackId'] == trackId] 
        count_length.append(len(user_traj))        
        ax.plot(user_traj['xCenter']/orthoPxToMeter, user_traj['yCenter']*-1/orthoPxToMeter, 
                color=color_dic[user_info[0, -1]] )

    plt.plot([], [], color='b', label='pedestrian') 
    plt.plot([], [], color='g', label='bicycle') 
    plt.plot([], [], color='r', label='car') 
    plt.plot([], [], color='c', label='truck_bus') 
        
    ax.set_title("%02.0f_Trajectories"%name)
    plt.legend()
    plt.gca().invert_yaxis()
    # plt.savefig("../plots/%02.0f_Trajectories"%name, dpi=400)
    plt.show()
    plt.gcf().clear()
    plt.close()
    
    
    count_length = np.asarray(count_length)
    print("mean length: %.1f, std: %.2f"%(np.mean(count_length), np.std(count_length)))
    
    
def get_seq_index(tracksMeta, dowmsample_rate, seq_length=20):
    '''
    This is the function to extract sequences have the minimum predefined length
    Like the trajnet, every road user will be extracted once
    It tries to keep all the trajectories in a scenario
    '''
    seq_index = []
    seq_length = seq_length*dowmsample_rate
    start_end = tracksMeta[['trackId', 'initialFrame', 'finalFrame']].values
    
    ini_start = np.min(start_end[:, 1])
    ini_end = ini_start+seq_length
        
    
    while not tracksMeta.empty:
        # ini_start = np.min(start_end[:, 1])
        # ini_end = ini_start+seq_length
        _tracksMeta = tracksMeta.loc[(tracksMeta['initialFrame']<=ini_start) 
                                & (tracksMeta['finalFrame']>=ini_end-1)]
        
        for i in _tracksMeta.trackId.unique():
            seq_index.append([i, ini_start, ini_end])
         
        # update the tracksMeta and remove the brocken ones    
        tracksMeta = tracksMeta.loc[tracksMeta['finalFrame']>=ini_end-1]
        tracksMeta = tracksMeta.drop(_tracksMeta.index.values)
        
        # update the initial start and end
        # ini_start = tracksMeta.initialFrame.min()
        # ini_start += seq_length/2 
        ini_start = ini_end
        ini_end = ini_start+seq_length
    
    seq_index = np.asarray(seq_index)    
    # print(seq_index)
    # print(len(seq_index))
    
    return seq_index
        
            
def extract_data(tracks, seq_index, dowmsample_rate=None):
    
    traj = []
    
    if dowmsample_rate:
        tracks = tracks.loc[tracks['frame']%dowmsample_rate==0]
    
    for index in seq_index:
        user_tracks = tracks.loc[tracks['trackId']==index[0]]
        user_tracks = user_tracks.loc[(user_tracks['frame']>=index[1])
                                      & (user_tracks['frame']<index[2])]
        user_tracks = user_tracks[['frame', 'trackId', 'xCenter', 'yCenter']]
        traj.append(user_tracks.values)
        
    traj = np.reshape(traj, [-1, 4])
    print(traj)
    return traj
        
    
            
        
    
    




if __name__ == "__main__":
    main()

