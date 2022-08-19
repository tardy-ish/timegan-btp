import os
import numpy as np
from PIL import Image

def import_sat(folder_path,n = -1):
    data = []
    i = 0
    for img_l in os.listdir(folder_path):
        if not img_l.endswith(".jpeg"):
            continue
        if n != -1 and i >= n:
            break
        i += 1
        image = Image.open(f"{folder_path}/{img_l}")
        image = cropImg(image)
        image = image.convert("L").resize((1024,1024))
        image = np.atleast_3d(image)
        data.append(image)
    return np.array(data)

def import_mnist(folder_path,n = -1):
    data = []
    i = 0
    for img_l in os.listdir(folder_path):
        if not img_l.endswith(".jpg"):
            continue
        if n != -1 and i >= n:
            break
        i += 1
        image = Image.open(f"{folder_path}/{img_l}")
        image = np.atleast_3d(image)
        data.append(image)
    data = np.array(data)
    print(data.shape)
    data = data.reshape((len(data),np.prod(data.shape[1:])))
    return data

def cropImg(image):
    left = 4
    top = 84
    right = left + 1200
    bottom = top + 1200
    return image.crop((left, top, right, bottom))

def cluster_data(data: np.array, seq_len):
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(data) - seq_len):
        _x = data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data