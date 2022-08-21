from math import ceil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os

def result_compile_mnist(data,fld,n,s):
    data = data*255.0
    if not os.path.exists(f"timeGAN_results/mnist/{fld}"):
        os.mkdir(f"timeGAN_results/mnist/{fld}")
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
            
def result_save(orig, enc, dec, img_size, dir_path,n = 5):
    for i in range(n):
        orig_img = orig[i].reshape((img_size,img_size))
        enc_img  = enc[i].reshape((img_size//16,img_size//8))
        dec_img  = dec[i].reshape((img_size,img_size))

        orig_img = Image.fromarray(orig_img)
        enc_img = Image.fromarray(enc_img)
        dec_img = Image.fromarray(dec_img)

        img_name = f"result-{i+1}"

        m_img = Image.new(mode="RGB",size=(860,360),color=(255,255,255))
        m_img.paste(orig_img.resize((256,256)),(24,50))
        m_img.paste(enc_img.resize((256,128)),(302,100))
        m_img.paste(dec_img.resize((256,256)),(580,50))
        fnt = ImageFont.truetype("FreeMono.ttf",size=20)
        txt = ImageDraw.Draw(m_img)
        txt.text((24+76,25),"original", fill=(0,0,0), font=fnt)
        txt.text((302+76,25),"encoded", fill=(0,0,0), font=fnt)
        txt.text((580+76,25),"decoded", fill=(0,0,0), font=fnt)
        txt.text((325,320),  img_name , fill=(0,0,0), font=fnt)
        m_img.save(f"{dir_path}/{img_name}.png")
