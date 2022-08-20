from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

from results_save import result_compile_mnist





def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    
    fld = f"./timeGAN_models/{args.mode}/{args.model}/timeGAN_model.pk1"
    if not path.exists(fld):
        print("Path doesn't exist, closing")
        return
    synth = TimeGAN.load(fld)
    synth_data = synth.sample(args.sample)
    print("Shape of Synthetic data:",synth_data.shape)
    n,s,_ = synth_data.shape
    result_compile_mnist(synth_data,args.model,n)


if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()

    #sample length, seq_len
    parser.add_argument(
        '--sample',
        help="length of samples",
        default=24,
        type=int)
    parser.add_argument(
        '--seq_len',
        help="length of sequence",
        default=24,
        type=int)
    
    #select gpu to use
    parser.add_argument(
        '--gpu_num',
        help='Select GPU',
        default=3,
        type=int)
    #model path
    parser.add_argument(
        '--model',
        help='folder where model is stored',
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
