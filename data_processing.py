import os
import numpy as np
from PIL import Image

def import_data(folder_path,n = -1):
    data = []
    i = 0
    for img_l in os.listdir(folder_path):
        if n != -1 and i >= n:
            break
        i += 1
        image = Image.open(f"{folder_path}/{img_l}").convert("L").resize((1024,1024))
        image = np.atleast_3d(image)
        data.append(image)
    return np.array(data)
