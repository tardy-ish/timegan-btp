from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os

def result_compile_mnist(data,fld,n):
    os.mkdir(f"timeGAN_results/mnist/{fld}")
    for k in range(n):
        bckg = Image.new(mode="RGB",size=(410,160),color=(255,255,255))
        y = 10
        for i in range(3):
            x = 10
            for j in range(8):
                img = data[k][i*8 + j].reshape((28,28))
                img = Image.fromarray(img)
                bckg.paste(img,(x,y))
                x += 50
            y += 50
        bckg.save(f"timeGAN_results/mnist/{fld}/{str(k+1).zfill(len(str(n)))}.png")
            

