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


def load_model(fld):
    pth = f"./autoencoder_models/{fld}"
    e = keras.models.load_module(f"{pth}/encoder.h5")
    d = keras.models.load_module(f"{pth}/decoder.h5")
    return e,d


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    
    # encoder,decoder = load_model(args.model)


    data = import_sat(args.image_path,1,args.img_count)

    # X_train, X_test,_, _ = train_test_split(data, data, test_size=0.05, random_state=42)
    # history = autoEncDec.fit(
    #     X_train,X_train, 
    #     epochs = args.epochs, 
    #     batch_size = args.batch_size, 
    #     validation_data=(X_test,X_test)
    # )

    # encoder,decoder = split_model(autoEncDec)

    # folder = save_model(encoder,decoder)

    # history_save(history.history,args.epochs,folder)
    
    # save_model(encoder,decoder)
    # print("Models have been saved")
    

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
    
    

    args = parser.parse_args() 

    # Calls main function  
    main(args)


