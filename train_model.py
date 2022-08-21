import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
IST = pytz.timezone('Asia/Kolkata')

from tensorflow import keras

import sys
sys.path.append('./ydata_synthetic')

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN

import argparse
from data_processing import cluster_data, import_sat, import_mnist

def save_model(model,mode):
    l = datetime.now(IST).strftime("%Y-%m-%d-%H-%M")
    
    new_dir = f"./timeGAN_models/{mode}/{l}"
    os.mkdir(new_dir)
    model.save(f"{new_dir}/timeGAN_model.pk1")

    return l

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    if args.mode == 'sat':
        data = import_sat(args.image_path,args.img_count)

        encoder = keras.models.load_model(f"./autoencoder_models/final/encoder.h5")
        decoder = keras.models.load_model(f"./autoencoder_models/final/decoder.h5")
        
        data = encoder.predict(data)

    elif args.mode == 'mnist':
        data = import_mnist(args.image_path,args.img_count)

    clust_data = cluster_data(data,args.seq_len)
    print(len(clust_data),clust_data[0].shape)
    n_seq = clust_data[0].shape[1]

    gan_args = ModelParameters(batch_size=args.batch_size,
                           lr=args.lr,
                           noise_dim=args.noise_dim,
                           layers_dim=args.dim)

    synth = TimeGAN(model_parameters=gan_args, hidden_dim=args.hidden_dim, seq_len=args.seq_len, n_seq=n_seq, gamma=args.gamma)
    synth.train(clust_data, train_steps=args.train_steps)
    t = save_model(synth,args.mode)
    print("Model trained and saved at:",t)

if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    #seq_len, train_steps, hidden_dim, gamma
    parser.add_argument(
        '--seq_len',
        help="length of sequence",
        default=24,
        type=int)
    parser.add_argument(
        '--train_steps',
        help='',
        default=500,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        help='',
        default=24,
        type=int)
    parser.add_argument(
        '--gamma',
        help='',
        default=1,
        type=int)

    #noise_dim, dim, batch_size    
    parser.add_argument(
        '--noise_dim',
        help='',
        default=32,
        type=int)
    parser.add_argument(
        '--dim',
        help='',
        default=128,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=8,
        type=int) 

    #log_step, learning_rate
    parser.add_argument(
        '--log_step',
        help='',
        default=100,
        type=int) 
    parser.add_argument(
        '--lr',
        help='',
        default=5e-4,
        type=float)
    
    #select gpu to use
    parser.add_argument(
        '--gpu_num',
        help='Select GPU',
        default=3,
        type=int)
    
    #image_path, image_count
    parser.add_argument(
        '--image_path',
        help="path to images",
        default=None,
        type=str)
    parser.add_argument(
        '--img_count',
        help='number of images to import',
        default=80,
        type=int)

    #training mode
    parser.add_argument(
        '--mode',
        choices=['mnist','sat'],
        default='mnist',
        type=str)

    args = parser.parse_args() 

    # Calls main function  
    main(args)
