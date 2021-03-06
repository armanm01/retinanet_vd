import os
from PIL import Image

#cropping with sliding window of size 1000x1000 and stride 800
def crop_by_1000(img):
    width, height = img.size
    cropped_imgs = []
    for j in range(0,height,800):
        for i in range(0, width, 800):
            if i+1000 >= width and j+1000 >= height:
                crop = img.crop((i,j,width,height))
                cropped_imgs.append(tuple((crop,i,j)))
            elif i+1000 >= width and j+1000 <height:
                crop = img.crop((i,j,width, j+1000))
                cropped_imgs.append(tuple((crop, i,j)))
            elif i+1000 < width and j+1000 >= height:
                crop = img.crop((i,j,i+1000,height))
                cropped_imgs.append(tuple((crop, i,j)))
            else:
                crop = img.crop((i,j,i+1000,j+1000))
                cropped_imgs.append(tuple((crop,i,j)))
    return cropped_imgs

#cropping for single image
crops = crop_by_1000(img)
for crop in crops:
    crop[0].save('-path-to-save-/{name}_{x}_{y}.png'.format(x=crop[1],y=crop[2],name=imgname), 'png')
