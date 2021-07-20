# show images inline
%matplotlib inline

# automatically reload modules when they have changed
%load_ext autoreload
%autoreload 2

# import keras
from tensorflow import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# use this to change which GPU to use
gpu = "0,"

# set the modified tf session as backend in keras
setup_gpu(gpu)

models_path = "C:/Users/arman/Desktop/Arman/cowc-everything/keras_fizyr_on_mycowc/Ineference_models/model_nms03_score02_max2000.h5"
model = models.load_model(models_path, backbone_name='resnet50')

labels_to_names = {0:"noncar", 1 : "car"}

# load image
path = ""

image = read_image_bgr(path)
print(image.shape)
# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

image, scale = resize_image(image,2000,2000)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if label == 1:
        # scores are sorted so we can break
        if score < 0.25:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        caption = "{:.3f}".format(score)
        draw_caption(draw, b,caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
