import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import data.h36m
import glob
import time
import matplotlib.patches as patches
import itertools
import spacepy



model_folder = '/home/jovyan/runs/metrabs-exp/tconv1_in160/h36m_METRO_m3s_rand1/model/'
model_detector_folder = "/home/jovyan/runs/metrabs-exp/models/eff2s_y4/model_multi"
input_image_size = 160
input_image = "/home/jovyan/data/inaki/inaki_moving1/frame1.jpg"
save_name ="test.jpg"



skeleton = [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12), 
            (7, 13), (13, 14), (14, 15), (7, 6), (6, 16), 
            (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]  # head




model_name = model_folder.split("/")
print(model_name[-3])

model_d = tf.saved_model.load(model_detector_folder)
print(model_d.detector.model.__dir__())
det = model_d.detector

model = tf.saved_model.load(model_folder)
print("Loading model is done")

img = Image.open(input_image)
bbox =  model_d.detector.predict_single_image(img) 

x, y, wd, ht, conf = bbox[0]
center = np.array([int(x + wd / 2), int(y + ht / 2)])
side_length = int(max(wd, ht) + 20)
crop = np.array(img)[center[1] - side_length // 2: center[1] + side_length // 2, center[0] - side_length // 2: center[0] + side_length // 2]
res = cv2.resize(crop, dsize=(input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)    


inp = res.astype(np.float16)
inp /= 256.
test = model(inp[np.newaxis,...], False)

xs = test.numpy()[0, :, 0]
ys = test.numpy()[0, :, 1]


i_root = -1                
test -= test[:, i_root, np.newaxis]

tt = test[0,:,:]
fig = plt.figure()

ax = fig.add_subplot(122, projection='3d')
tt = tt.numpy()
ax.scatter(tt[:, 0], -tt[:, 2], -tt[:, 1], s=1, c='r')

for i, j in skeleton:
    plt.plot([tt[i, 0], tt[j, 0]], [-tt[i, 2], -tt[j, 2]], [-tt[i, 1], -tt[j, 1]], 'r')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim3d(-700, 700)
ax.set_zlim3d(-700, 700)
ax.set_ylim3d(-700, 700)

ax2 = fig.add_subplot(121)
ax2.imshow(img)
rect = patches.Rectangle((x, y), wd, ht, linewidth=1, edgecolor='r', facecolor='none')
ax2.add_patch(rect)
plt.savefig(save_name)
plt.close()



    # k=k+1

