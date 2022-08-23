import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import argparse
from datetime import datetime
import pytz
import csv
IST = pytz.timezone('Asia/Kolkata')

from data_processing import import_sat
from data_processing import get_scale
from results_save import enc_result



def load_model(fld):
    pth = f"./autoencoder_models/{fld}"
    e = keras.models.load_model(f"{pth}/encoder.h5",compile=False)
    d = keras.models.load_model(f"{pth}/decoder.h5",compile=False)
    return e,d

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    
    encoder,decoder = load_model(args.model)
    scale = get_scale(encoder,1)

    data = import_sat(args.image_path,args.sample,scale)
    enc_data = encoder.predict(data)
    dec_data = decoder.predict(enc_data)

    # print(data.shape,enc_data.shape,dec_data.shape)
    enc_result(data,enc_data,dec_data,scale,args.model,args.sample)    

if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        help="path to images",
        default=None,
        type=str)
    parser.add_argument(
        '--sample',
        help='number of results to print out',
        default=10,
        type=int)
    parser.add_argument(
        '--model',
        help='folder where model is stored',
        default=None,
        type=str)    
    parser.add_argument(
        '--gpu_num',
        help='Select GPU',
        default=3,
        type=int)
    
    
    

    args = parser.parse_args() 

    # Calls main function  
    main(args)


