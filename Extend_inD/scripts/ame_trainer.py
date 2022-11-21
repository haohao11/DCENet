# -*- coding: utf-8 -*-
"""

@author: cheng
"""
import argparse
import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import sys
import time

from collision import check_collision
from datainfo import datainfo
from dataloader import preprocess_data, loaddata
from evaluation import get_errors
from ame_model import Acvae
from mkdir import mak_dir
from plots import plot_pred
import writer
from ranking import gauss_rank
import pathlib

np.set_printoptions(suppress=True)


def main():
    
    desc = "Keras implementation of CVAE for trajectory prediction"
    parser = argparse.ArgumentParser(description=desc) 
        
    parser.add_argument('--num_pred', type=int, default=10, help='This is the number of predictions for each agent')
    parser.add_argument('--obs_seq', type=int, default=8, help='Number of time steps observed')
    parser.add_argument('--enviro_pdim', type=int, default=[32, 32, 3], help='The dimension of the environment after padding')
    parser.add_argument('--pred_seq', type=int, default=12, help='Number of time steps to be predicted')    
    parser.add_argument('--dist_thre', type=float, default=1.0, help='The distance threhold for group detection')
    parser.add_argument('--ratio', type=float, default=0.95, help='The overlap ratio of coexisting for group detection')   
    parser.add_argument('--n_hidden', type=int, default=1024, help='This is the hidden size of the cvae') 
    parser.add_argument('--z_dim', type=int, default=2, help='This is the size of the latent variable')
    parser.add_argument('--encoder_dim', type=int, default=16, help='This is the size of the encoder output dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=64, help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='The size of GRU hidden state')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--o_drop', type=float, default=0.85, help='The dropout rate for occupancy')
    parser.add_argument('--s_drop', type=float, default=0.0, help='The dropout rate for trajectory sequence')
    parser.add_argument('--z_drop', type=float, default=0.15, help='The dropout rate for z input')
    parser.add_argument('--beta', type=float, default=0.65, help='Loss weight')
    parser.add_argument('--query_dim', type=int, default=2, help='The dimension of the query')
    parser.add_argument('--keyvalue_dim', type=int, default=2, help='The dimension for key and value')    
    parser.add_argument('--train_mode', action='store_false', default=True, help='This is the training mode')
    parser.add_argument('--train_set', type=str, choices=['Train'], default='sharedspaces', 
                        help='This is the directories for the training data')
    parser.add_argument('--challenge_set', type=str, choices=['Test'], default='Test', 
                        help='This is the directories for the challenge data')
    parser.add_argument('--split', type=float, default=0.8, help='the split rate for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # train with 3e-4, fine-tune with 1e-4
    parser.add_argument('--aug_num', type=int, default=1, help='Number of augmentations')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of batches')
    parser.add_argument('--patience', type=int, default=30, help='Maximum mumber of continuous epochs without converging')    
    args = parser.parse_args(sys.argv[1:])

    # specify which GPU(s) to be used, gpu device starts from 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Use the default CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Make all the necessary folders 
    mak_dir()
    

    # # specify the directory for training and challenge data   
    train_paths= sorted(glob.glob("../inD-dataset-v1.0/trajectories/*.txt"))
    
         
    # Process the data
    for path in train_paths:
        # dataname = path.split('\\')[-1].split('.')[0]
        dataname = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(pathlib.Path(__file__).parent / f"../processed_data/train/{dataname}.npz"):
            # preprocess_data(path, args.obs_seq+args.pred_seq-1, args.enviro_pdim, "train")            
            preprocess_data(seq_length=args.obs_seq+args.pred_seq-1,
                            size=args.enviro_pdim,
                            dirname="train",
                            path=path,
                            aug_num=args.aug_num,
                            save=True)
       
            
    # Check the datainfo for dataset partition        
    Datalist = datainfo(args.aug_num)
    
    # Define the callback and early stop
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath="../models/ame_%s_%0.f_%s.hdf5"%(str(args.pred_seq), args.epochs, timestr)
    ## Eraly stop
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]  
  
    # # Instantiate the model
    acvae = Acvae(args)   
    # Contruct the cave model    
    train =acvae.training() 
    train.summary()
        
    # Start training phase
    if args.train_mode: 
        # # traindata_list = Datalist.train_crowds
        # # traindata_list = Datalist.train_data
        # traindata_list = Datalist.train_merged
        # print("traindata_list", traindata_list)
                
        train_merged_exist = os.path.join("../processed_data/train", "train_merged.npz")            
        if not os.path.exists(pathlib.Path(__file__).parent / train_merged_exist):
            traindata_list = Datalist.train_data               
        else:
            traindata_list = Datalist.train_merged
        print("traindata_list", traindata_list)
        
        # Get the data fro training andvalidation
        np.random.seed(10)         
        offsets, traj_data, occupancy = loaddata(traindata_list, args, datatype="train")                   
        train_val_split = np.random.rand(len(offsets)) < args.split
        
        train_x = offsets[train_val_split, :args.obs_seq-1, 4:6]
        train_occu = occupancy[train_val_split, :args.obs_seq-1, ..., :args.enviro_pdim[-1]]        
        train_y = offsets[train_val_split, args.obs_seq-1:args.obs_seq-1+args.pred_seq, 4:6]
        train_y_occu = occupancy[train_val_split, args.obs_seq-1:args.obs_seq-1+args.pred_seq, ..., :args.enviro_pdim[-1]]
        
        val_x = offsets[~train_val_split, :args.obs_seq-1, 4:6]
        val_occu = occupancy[~train_val_split, :args.obs_seq-1, ..., :args.enviro_pdim[-1]]
        val_y = offsets[~train_val_split, args.obs_seq-1:args.obs_seq-1+args.pred_seq, 4:6]
        val_y_occu = occupancy[~train_val_split, args.obs_seq-1:args.obs_seq-1+args.pred_seq, ..., :args.enviro_pdim[-1]]
        
        print("%.0f trajectories for training\n %.0f trajectories for valiadation"%
              (train_x.shape[0], val_x.shape[0]))
        
        print("Start training the model...") 
        # Retrain from last time
        # train.load_weights("../models/ame_12_1000_20220923-013958.hdf5")         
        train.fit(x=[train_occu, train_x, train_y_occu, train_y],
                      y=train_y,
                      shuffle=True,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=([val_occu, val_x, val_y_occu, val_y], val_y))
        train.load_weights(filepath)
        
        
    else:
        print('Run pretrained model')
        train.load_weights(pathlib.Path(__file__).parent / "../models/ame_12_1000_20220923-013958.hdf5")
        
        
    for i in range(len(Datalist.test_data)):
        testdata_list = Datalist.test_data[i:i+1]        
        print("\n\nTest on %s"%(testdata_list[0]))
        
        test_offsets, test_trajs, test_occupancy = loaddata(testdata_list, args, datatype="test") 
        
        test_x = test_offsets[:, :args.obs_seq-1, 4:6]    
        test_occu = test_occupancy[:, :args.obs_seq-1, ..., :args.enviro_pdim[-1]]
        # last_obs_test = test_offsets[:, args.obs_seq-2, 2:4]
        last_obs_test = test_trajs[:, args.obs_seq-1, 2:4]
        y_truth = test_trajs[:, args.obs_seq:args.obs_seq+args.pred_seq, :4]
        xy_truth = test_trajs[:, :args.obs_seq+args.pred_seq, :4]       
        print("%.0f trajectories for testing"%(test_x.shape[0]))
      
        # Start inference phase      
        # Retrieve the x_encoder and the decoder   
        x_encoder=acvae.X_encoder()
        decoder = acvae.Decoder()       
        # x_encoder.summary()
        # decoder.summary()
                
        # get the x_encoded_dense as latent feature for prediction
        x_latent = x_encoder.predict([test_occu, test_x], batch_size=args.batch_size)
                
        # Using x_latent and z as input of the decoder for generating future trajectories
        print("Start predicting")
        predictions = []
        for i, x_ in enumerate(x_latent):
            last_pos = last_obs_test[i]
            x_ = np.reshape(x_, [1, -1])
            for i in range(args.num_pred):
                # sampling z from a normal distribution
                z_sample = np.random.rand(1, args.z_dim)
                y_p = decoder.predict(np.column_stack([z_sample, x_]))
                y_p_ = np.concatenate(([last_pos], np.squeeze(y_p)), axis=0)
                y_p_sum = np.cumsum(y_p_, axis=0)
                predictions.append(y_p_sum[1:, :])
        predictions = np.reshape(predictions, [-1, args.num_pred, args.pred_seq, 2])
            
        print('Predicting done!')
        print(predictions.shape)    
        #plot_pred(xy_truth, predictions)    
        # Get the errors for ADE, DEF, Hausdorff distance, speed deviation, heading error
        print("\nEvaluation results @top%.0f"%args.num_pred)
        errors = get_errors(y_truth, predictions)
        collision = check_collision(y_truth, obs_seq=args.obs_seq, pred_seq=args.pred_seq)
        sta_top = np.hstack((errors, collision))
        
        
        ##        
        ## Get the first time prediction      
        ranked_prediction = []
        for prediction in predictions:
            ranks = gauss_rank(prediction)
            ranked_prediction.append(prediction[np.argmax(ranks)])
        ranked_prediction = np.reshape(ranked_prediction, [-1, 1, args.pred_seq, 2])
        print("\nEvaluation results for most-likely predictions")
        ranked_errors = get_errors(y_truth, ranked_prediction)
                
        first_pred_traj = writer.get_index(test_trajs[:, :args.obs_seq, :], predictions)
        print("Collision in ranked prediction")     
        ranked_collision = check_collision(np.squeeze(first_pred_traj), obs_seq=args.obs_seq, pred_seq=args.pred_seq)
        #writer.write_pred_txt(test_trajs[:, :args.obs_seq, :], predictions, str(args.pred_seq)+"_"+testdata_list[0], "results/AMENet/prediction_first")        
        writer.write_pred_txt(test_trajs[:, :args.obs_seq, :], predictions, str(args.pred_seq)+"_"+testdata_list[0], "results/AMENet")        
        sta_ranked = np.hstack((ranked_errors, ranked_collision))
        
        
if __name__ == "__main__":
    main()
