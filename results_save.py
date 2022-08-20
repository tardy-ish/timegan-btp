from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from math import ceil

def result_compile_mnist(data,fld,n,s):
    if not os.path.exists(f"timeGAN_results/mnist/{fld}"):
        os.mkdir(f"timeGAN_results/mnist/{fld}")    
    print(data.shape)
    for k in range(n):
        bckg = Image.new(mode="RGB",size=(410,160),color=(255,255,255))
        y = 10
        for i in range(ceil(s/8)):
            x = 10
            k = False
            for j in range(8):
                if i*8 + j >= s:
                    k = True
                    break
                # print(i*8 + j)
                img = data[k][i*8 + j].reshape((28,28))
                img = Image.fromarray(img)
                bckg.paste(img.resize((40,40)),(x,y))
                x += 50
                if k:
                    break
            y += 50
        bckg.save(f"timeGAN_results/mnist/{fld}/{str(k+1).zfill(len(str(n)))}.png")
            

