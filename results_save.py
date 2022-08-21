from math import ceil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os

def result_compile_mnist(data,fld,n,s):
    data = data*255.0
    if not os.path.exists(f"timeGAN_results/mnist/{fld}"):
        os.mkdir(f"timeGAN_results/mnist/{fld}")
    k = 0
    for k in range(n//3):
        bckg = Image.new(mode="RGB",size=(490,250),color=(255,255,255))
        y = 10
        for j in range(3):
            x = 10
            for i in range(s):
                img = data[k*3 + j][i].reshape((28,28))
                img = Image.fromarray(img)
                bckg.paste(img.resize((70,70)),(x,y))
                x += 80
            y += 80
        bckg.save(f"timeGAN_results/mnist/{fld}/{str(k+1).zfill(len(str(n)))}.png")
            
