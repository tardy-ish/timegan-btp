import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow import keras
import argparse
from datetime import datetime
import pytz
IST = pytz.timezone('Asia/Kolkata')

from data_processing import import_data

def split_model(model_s):
    encoder = keras.Model(model_s.get_layer("INPUT").input,model_s.get_layer("CODE").output)
    code_input_layer = keras.Input(shape=(32, 32, 8), name="CODE_INPUT")
    x = code_input_layer
    for lay in model_s.layers[11:]:
        x = lay(x)
    decoder = keras.Model(code_input_layer,x)
    return encoder,decoder

def create_model(image_shape):
    input_layer = keras.Input(shape=image_shape, name="INPUT")
    x = keras.layers.Conv2D(16, (7,7), activation='relu', padding='same')(input_layer)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(8, (9,9), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(8, (11,11), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(8, (9,9), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(8, (7,7), activation='relu', padding='same')(x)

    code_layer = keras.layers.MaxPooling2D((2, 2), name="CODE")(x)

    x = keras.layers.Conv2DTranspose(8, (7,7), activation='relu', padding='same')(code_layer)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(8, (9,9), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(8, (11,11), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(8, (9,9), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2DTranspose(16, (7,7), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2,2))(x) 
    x = keras.layers.Dropout(0.5)(x)
    output_layer = keras.layers.Conv2D(1, (7,7), padding='same', name="OUTPUT")(x)

    return keras.Model(input_layer,output_layer)

def save_model(enc,dec):
    l = datetime.now(IST).strftime("%Y-%m-%d-%H-%M")
    cwd = os.getcwd()
    new_dir = f"{cwd}/autoencoder_models/{l}"
    os.mkdir(new_dir)
    
    enc.save(f"{new_dir}/encoder.h5")
    dec.save(f"{new_dir}/decoder.h5")

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    
    data = import_data(args.image_path,args.img_count)
    
    image_shape = data.shape[1:]
    autoEncDec = create_model(image_shape)
    autoEncDec.compile(optimizer='adam', loss=args.loss)
    autoEncDec.summary()

    X_train, X_test,_, _ = train_test_split(data, data, test_size=0.05, random_state=42)
    history = autoEncDec.fit(
        X_train,X_train, 
        epochs = args.epochs, 
        batch_size = args.batch_size, 
        validation_data=(X_test,X_test)
    )
    print(history.history)
    encoder,decoder = split_model(autoEncDec)

    save_model(encoder,decoder)
    print("Models have been saved")
    
# py autoEncDec.py --image_path /workspace/timegan-btp/data/images/ --img_count -1 --epochs 300 --batch_size 8 --gpu_num 3
if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '--epochs',
        help='the number of epochs (should be optimized)',
        default=50,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=8,
        type=int)    
    parser.add_argument(
        '--loss',
        choices=['mse','binary_crossentropy','mean_squared_logarithmic_error'],
        default='mean_squared_logarithmic_error',
        type=str)
    parser.add_argument(
        '--gpu_num',
        help='Select GPU',
        default=3,
        type=int)
    
    

    args = parser.parse_args() 

    # Calls main function  
    main(args)


