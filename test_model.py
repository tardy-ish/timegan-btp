import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

import argparse
from data_processing import import_data

CWD = os.getcwd()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    data = import_data(args.image_path,args.img_count)

    encoder = keras.models.load_model(f"{CWD}/autoencoder_models/encoder.h5")
    decoder = keras.models.load_model(f"{CWD}/autoencoder_models/decoder.h5")

    enc_data = encoder.predict(data)

    gan_args = ModelParameters(batch_size=args.batch_size,
                           lr=args.lr,
                           noise_dim=args.noise_dim,
                           layers_dim=args.dim)




if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    #seq_len, n_seq, hidden_dim, gamma
    parser.add_argument(
        '--seq_len',
        help="length of sequence",
        default=24,
        type=int)
    parser.add_argument(
        '--n_seq',
        help='',
        default=6,
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

    args = parser.parse_args() 

    # Calls main function  
    main(args)