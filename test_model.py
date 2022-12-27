from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append('./ydata_synthetic')

from ydata_synthetic.synthesizers.timeseries import TimeGAN
from tensorflow import keras

from data_processing import get_scale
from results_save import result_compile_mnist, timegan_sat

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    fld = f"./timeGAN_models/{args.mode}/{args.model_time}/timeGAN_model.pk1"
    if not path.exists(fld):
        print("Path doesn't exist, closing")
        return
    synth = TimeGAN.load(fld)
    if args.mode == 'sat':
        decoder = keras.models.load_model(f"./autoencoder_models/{args.model_auto}/decoder.h5",compile=False)
    synth_data = synth.sample(args.seed)
    print("Shape of Synthetic data:",synth_data.shape)
    n,s,_ = synth_data.shape
    if args.mode == "mnist":
        result_compile_mnist(synth_data,args.model,n,s)
    elif args.mode == "sat":
        scale = get_scale(decoder,-1)
        dim = 32//scale
        synth_data = synth_data.reshape((n,s,dim,dim,8))
        synth_data = decoder.predict(synth_data)
        timegan_sat(synth_data,args.model,n,s)


if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    #sample length
    parser.add_argument(
        '--seed',
        help="value which affects the sample generation",
        default=24,
        type=int)
    
    #select gpu to use
    parser.add_argument(
        '--gpu_num',
        help='Select GPU',
        default=3,
        type=int)

    #model paths
    parser.add_argument(
        '--model_time',
        help='folder where timegan model is stored',
        default=None,
        type=str)
    parser.add_argument(
        '--model_auto',
        help='folder where encoder decoder models are stored',
        default=None,
        type=str)

    #testing mode
    parser.add_argument(
        '--mode',
        choices=['mnist','sat'],
        default='mnist',
        type=str)

    args = parser.parse_args() 

    # Calls main function  
    main(args)
