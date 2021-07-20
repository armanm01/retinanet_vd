import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
%matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil

def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename','xmin', 'ymin', 'xmax', 'ymax','class']
                             )
    grouped = df.groupby('filename')
    
    for filename in df['filename'].unique():
    #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)   
    #   read the image
        image = imageio.imread(images_path+filename)
    #   get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'class'], axis=1).values
    #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    #   disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
    #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
    #   don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        
    #   otherwise continue
        else:
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   rename filenames by adding the predifined prefix
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])            
    
    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy





# augmentation methods – rotation, flips and translations  
aug_list = [iaa.Affine(rotate=15),iaa.Affine(rotate=30),iaa.Affine(rotate=45),iaa.Affine(rotate=60),
           iaa.Affine(rotate=75),iaa.Affine(rotate=90),iaa.Affine(rotate=105),iaa.Affine(rotate=120),
           iaa.Affine(rotate=135),iaa.Affine(rotate=150),iaa.Affine(rotate=165),
           iaa.Fliplr(1),iaa.Flipud(1),
           iaa.TranslateX(percent=(-0.2)),
           iaa.TranslateX(percent=(0.2)),
           iaa.TranslateY(percent=(-0.2)),
           iaa.TranslateY(percent=(0.2))]

# applying every augmentation to every cropped train image 
for i in range(len(aug_list)):
    augmented_images_df_list.append(image_aug(df, 'C:/Users/arman/Desktop/Arman/cowc-everything/keras_fizyr_on_mycowc/keras-retinanet-0.5.1/train/', 
                                    'C:/Users/arman/Desktop/Arman/cowc-everything/keras_fizyr_on_mycowc/keras-retinanet-0.5.1/train/aug_images/', 
                                    'aug_{num}_'.format(num=i), aug_list[i]))
