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
import sys


def load_coords(path):
    i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    with spacepy.pycdf.CDF(path) as cdf_file:
        coords_raw = np.array(cdf_file['Pose'], np.float32)[0]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    return coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]


skeleton = [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12), 
            (7, 13), (13, 14), (14, 15), (7, 6), (6, 16), 
            (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]  # head

# usage: python viz.py model_path 
data_root = '/home/jovyan/data/metrabs-processed/h36m/'
exp_root = '/home/jovyan/runs/metrabs-exp/'

#model_folder = exp_root + 'tconv1_in112/h36m_METRO_m3s03_rand1_1500/model/'
model_folder = [exp_root + p + '/model/' for p in sys.argv[1:-1]]
for p in model_folder:
    if not os.path.isdir(p):
        raise OSError("model_folder not exist" + p)

model_name = [p.split("/") for p in model_folder]
input_size = [112 if "112" in p[-4] else 160 if "160" in p[-4] else 256 for p in model_name]
print(model_name[:][-4:-2], input_size)

model_detector_folder = exp_root + "models/eff2s_y4/model_multi"
model_d = tf.saved_model.load(model_detector_folder)
det = model_d.detector

model = [tf.saved_model.load(p) for p in model_folder]
print("Loading model is done")

#for filename in glob.glob('Directions/*.jpg'): #assuming gif


camera_names = ['55011271', '54138969', '58860488', '60457274']
frame_step = 5
all_world_coords = []
all_image_relpaths = []

for i_subj in [11]: # [9, 11]:
    number_activity=0    
    for activity, cam_id in itertools.product(data.h36m.get_activity_names(i_subj), range(4)):    
        # print(activity, cam_id, number_activity)
               
            if not 'Directions' in activity: # number_activity==30:
                continue
            camera_name = camera_names[cam_id]
            pose_folder = data_root + f'S{i_subj}/MyPoseFeatures'
            coord_path = f'{pose_folder}/D3_Positions/{activity}.cdf'
            world_coords = load_coords(coord_path)
            n_frames_total = len(world_coords)       
            image_relfolder = data_root + f'S{i_subj}/Images/{activity}.{camera_name}'

            MYDIR = (exp_root + "visualize/" + sys.argv[-1] + f'/S{i_subj}_{activity}.{camera_name}/' )

            CHECK_FOLDER = os.path.isdir(MYDIR)
            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)
            else:
                print(MYDIR, "folder already exists.")    
                
            total_frame = n_frames_total//5

            out_test = np.zeros((total_frame,17,3))
            k=0
            for i_frame in range(0, n_frames_total, frame_step):

                image_path = image_relfolder +"/" + str(i_frame)
                path_parts = image_path.split('/')
                last_part = path_parts[-1].split(' ')
                save_path = MYDIR + f'frame_{int(last_part[-1]):06}.jpg'

                last_part[-1] = f'frame_{int(last_part[-1]):06}'
                path_parts[-1] = ' '.join(last_part)
                image_path = '/'.join(path_parts) +'.jpg'

                img = Image.open(image_path)
                bbox =  model_d.detector.predict_single_image(img) 

                x, y, wd, ht, conf = bbox[0]
                crop = np.array(img)[int(y):int(y+ht), int(x):int(x+wd)]
                pred = []
                for i_m, mod in enumerate(model):
                    res = cv2.resize(crop, dsize=(input_size[i_m], input_size[i_m]), interpolation=cv2.INTER_CUBIC)    

                    inp = res.astype(np.float16)
                    inp /= 256.
                    test = mod(inp[np.newaxis,...], False)

                    xs = test.numpy()[0, :, 0]
                    ys = test.numpy()[0, :, 1]
                
                    raw_output = test[0,:,:]
                    i_root = -1                
                    test -= test[:, i_root, np.newaxis]

                    tt = test[0,:,:]
                    pred.append(tt)

    #            np.save("test", tt)
                deg = 5
                views = [(deg, deg-90), (deg, deg), (90-deg, deg-90)]
                fsz = 2
                fig = plt.figure(figsize=(fsz*len(model)+fsz, fsz*len(views)))
                # Add a 3D subplot

                for i_m in range(len(model)):
                    tt = pred[i_m]
                    tt = tt.numpy()
                    #    tt = tt[[3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 16], :]

                    for i_v in range(len(views)):
                        ax = fig.add_subplot(len(views),len(model)+1,(len(model)+1)*i_v + i_m+2, projection='3d')
                        ax.view_init(*views[i_v])
                        ax.scatter(tt[:, 0], -tt[:, 2], -tt[:, 1], s=1, c='r')

                        for i, j in skeleton:
                            plt.plot([tt[i, 0], tt[j, 0]], [-tt[i, 2], -tt[j, 2]], [-tt[i, 1], -tt[j, 1]], 'r')

                        #ax.set_xlabel('x')
                        #ax.set_ylabel('y')
                        #ax.set_zlabel('z')

                        ax.set_xlim3d(-700, 700)
                        ax.set_zlim3d(-700, 700)
                        ax.set_ylim3d(-700, 700)
                        ax.set_title(f"{model_name[i_m][-3].split('_')[2]}_in{input_size[i_m]}")

                ax2 = fig.add_subplot(len(views),len(model)+1,1)
                ax2.imshow(img)
                rect = patches.Rectangle((x, y), wd, ht, linewidth=1, edgecolor='r', facecolor='none')
                ax2.add_patch(rect)
                ax2.set_title(f'S{i_subj}_{activity}_{cam_id}')

                plt.axis('off')
                plt.savefig(save_path)

                plt.close()
                
                #out_test[k,:,:] = raw_output[:,:]
                #k=k+1

            break
            #np.save(save_path)                
            number_activity =  number_activity+1
