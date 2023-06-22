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
import json


def load_coords(path):
    i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    with spacepy.pycdf.CDF(path) as cdf_file:
        coords_raw = np.array(cdf_file['Pose'], np.float32)[0]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    return coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]


def plot_skeleton(axis, p, color='r'):
    axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color)

    skeleton = [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12),
            (7, 13), (13, 14), (14, 15), (7, 6), (6, 16), 
            (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]  # head

    for i, j in skeleton:
        plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], [p[i, 2], p[j, 2]], color)

        

def upper_plot_skeleton(axis, p, color='r'):
    axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color)

    skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1,4)]  # head
    axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color, alpha=1.0 )

    for i, j in skeleton:
        plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], [p[i, 2], p[j, 2]], color)            

    center_shoulder = (p[1,:] + p[4,:])/2.0     
    plt.plot([center_shoulder[0], p[0, 0]], [center_shoulder[1], p[0, 1]], [center_shoulder[2], p[0, 2]], color)            
    plt.plot([center_shoulder[0], p[7, 0]], [center_shoulder[1], p[7, 1]], [center_shoulder[2], p[7, 2]], color)           
        


def get_crop(img0, x0, y0, w0, h0):
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x0 + w0 > img0.width: w0 = img0.width - x0
    if y0 + h0 > img0.height: h0 = img0.height - y0
    return np.array(img0)[int(y0):int(y0 + h0), int(x0):int(x0 + w0)]


def get_pred(img_crop, in_size, mdl, camera_rotate=None):
    crop_resize = cv2.resize(img_crop, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
    inp = crop_resize.astype(np.float16) / 256.
    test = mdl(inp[np.newaxis, ...], False)
    test -= test[:, -1, np.newaxis]
    tt = test[0, :, :]  # (1,17,3)  cam.R (1,3,3) cam.t (1,3,1)
    return tt @ camera_rotate[0].T if camera_rotate is not None else tt, crop_resize


def prep_dir(target_path):
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
        print("created folder : ", target_path)
    else:
        print(target_path, "folder already exists.")


# usage: python viz.py model_path1 .. model_pathN ('h36m' or input_path) outname
args = sys.argv[1:]
outname = args[-1]
input_path = args[-2]
model_folders = args[:-2]

exp_root = '/home/jovyan/runs/metrabs-exp/'

#model_folder = exp_root + 'tconv1_in112/h36m_METRO_m3s03_rand1_1500/model/'
model_folder = [exp_root + p + '/model/' for p in model_folders]
for p in model_folder:
    if not os.path.isdir(p):
        raise OSError("model_folder not exist" + p)

model_name = [p.split("/") for p in model_folder]
input_size = [112 if "112" in p else 160 if "160" in p else 256 for p in model_folders]
print(model_name, input_size)

model_detector_folder = exp_root + "models/eff2s_y4/model_multi"
model_d = tf.saved_model.load(model_detector_folder)
det = model_d.detector

models = [tf.saved_model.load(p) for p in model_folder]
nmdl = len(models)
print("Loading model is done")


frame_step = 5
deg = 5
views = [(deg, deg - 90), (deg, deg), (90 - deg, deg - 90)]
fsz = 2

if input_path.startswith('h36m'):
    data_root = '/home/jovyan/data/metrabs-processed/h36m/'

    cam_param = exp_root + "visualize/human36m-camera-parameters/camera-parameters.json"
    with open(cam_param, 'r') as j:
         cam = json.loads(j.read())

    camera_names = ['55011271', '54138969', '58860488', '60457274']
    all_world_coords = []
    all_image_relpaths = []

    for i_subj in [9, 11]: # [9, 11]:
        number_activity=0
        for activity, cam_id in itertools.product(data.h36m.get_activity_names(i_subj), range(4)):
                print(activity, cam_id, number_activity)
                im_arr = []

                if len(input_path) > 4 and not input_path[4:] in activity:
                    continue
                cam_p = cam['extrinsics']['S'+str(i_subj)][camera_names[cam_id]]
                cam_rot = np.expand_dims(np.linalg.inv(np.array(cam_p['R'])), axis=0)  # (1,3,3)
                cam_loc = np.expand_dims(np.array(cam_p['t'])[:,0], axis=0)            # (1,3)

                camera_name = camera_names[cam_id]
                pose_folder = data_root + f'S{i_subj}/MyPoseFeatures'
                coord_path = f'{pose_folder}/D3_Positions/{activity}.cdf'
                world_coords = load_coords(coord_path)
                world_coords -= world_coords[:, -1, np.newaxis]

                n_frames_total = len(world_coords)
                image_relfolder = data_root + f'S{i_subj}/Images/{activity}.{camera_name}'

                MYDIR = (exp_root + "visualize/" + outname + f'/S{i_subj}_{activity}.{camera_name}/')

                prep_dir(MYDIR)
                '''
                CHECK_FOLDER = os.path.isdir(MYDIR)
                if not CHECK_FOLDER:
                    os.makedirs(MYDIR)
                    print("created folder : ", MYDIR)
                else:
                    print(MYDIR, "folder already exists.")
                '''

                total_frame = n_frames_total//5

                out_test = np.zeros((total_frame,17,3))
                k=0
                bboxes_gt = np.load(f'{data_root}/S{i_subj}/BBoxes/{activity}.{camera_name}.npy')

                for i_frame in range(0, n_frames_total, frame_step):

                    image_path = image_relfolder +"/" + str(i_frame)
                    path_parts = image_path.split('/')
                    last_part = path_parts[-1].split(' ')
                    save_path = MYDIR + f'frame_{int(last_part[-1]):06}.jpg'

                    last_part[-1] = f'frame_{int(last_part[-1]):06}'
                    path_parts[-1] = ' '.join(last_part)
                    image_path = '/'.join(path_parts) +'.jpg'

                    img = Image.open(image_path)
                    bbox = model_d.detector.predict_single_image(img)

                    x, y, wd, ht, conf = bbox[0]
                    if wd < ht:
                        y_sq, ht_sq = y, ht
                        x_sq, wd_sq = x-(ht-wd)/2., ht
                    else:
                        x_sq, wd_sq = x, wd
                        y_sq, ht_sq = y-(wd-ht)/2., wd
                    x_gt, y_gt, wd_gt, ht_gt = bboxes_gt[i_frame]

                    crop = get_crop(img, x, y, wd, ht)
                    crop_sq = get_crop(img, x_sq, y_sq, wd_sq, ht_sq)
                    crop_gt = get_crop(img, x_gt, y_gt, wd_gt, ht_gt)
                    pred_w, pred_w_gt, pred_w_sq, gt_w = [], [], [], world_coords[i_frame]
                    for i_m, model in enumerate(models):
                        tw, res = get_pred(crop, input_size[i_m], model, cam_rot)
                        tw_bb, res_bb = get_pred(crop_gt, input_size[i_m], model, cam_rot)
                        tw_sq, res_sq = get_pred(crop_sq, input_size[i_m], model, cam_rot)
                        pred_w.append(tw)
                        pred_w_gt.append(tw_bb)
                        pred_w_sq.append(tw_sq)

                    fig = plt.figure(figsize=(fsz*nmdl+fsz, fsz*len(views)))
                    # Add a 3D subplot

                    for i_m in range(nmdl):                        
                        #tt = pred[i_m].numpy()
                        tw = pred_w[i_m].numpy()
                        tw_bb = pred_w_gt[i_m].numpy()
                        tw_sq = pred_w_sq[i_m].numpy()
                        plot_fn = plot_skeleton if len(tw_sq) > 8 else upper_plot_skeleton

                        for i_v in range(len(views)):
                            ax = fig.add_subplot(len(views),nmdl+1,(nmdl+1)*i_v + i_m+2, projection='3d')
                            ax.view_init(*views[i_v])
                            plot_skeleton(ax, gt_w, 'b')
                            plot_fn(ax, tw_sq, 'r')

                            #ax.set_xlabel('x')
                            ax.set_xlim3d(-700, 700)
                            ax.set_zlim3d(-700, 700)
                            ax.set_ylim3d(-700, 700)
                            if i_v == 0:
                                ax.set_title(f"{model_name[i_m][-3].split('_')[2]}_in{input_size[i_m]}")

                    ax2 = fig.add_subplot(len(views),nmdl+1,1)
                    ax2.imshow(img)
                    rect = patches.Rectangle((x, y), wd, ht, linewidth=1, edgecolor='r', facecolor='none')
                    rect_gt = patches.Rectangle((x_gt, y_gt), wd_gt, ht_gt, linewidth=1, edgecolor='g', facecolor='none')
                    rect_sq = patches.Rectangle((x_sq, y_sq), wd_sq, ht_sq, linewidth=1, edgecolor='b', facecolor='none')
                    ax2.add_patch(rect_sq)
                    ax2.add_patch(rect)
                    ax2.add_patch(rect_gt)
                    ax2.set_title(f'S{i_subj}_{activity}_{cam_id}')

                    ax3 = fig.add_subplot(len(views),nmdl+1,nmdl+2)
                    ax3.imshow(res)

                    ax4 = fig.add_subplot(len(views),nmdl+1,2*nmdl+3)
                    ax4.imshow(res_sq)

                    plt.axis('off')
                    plt.savefig(save_path)

                    plt.close()

                    img = cv2.imread(save_path)
                    h, w, c = img.shape
                    im_arr.append(img)

                out = cv2.VideoWriter(exp_root + "visualize/" + outname + f'_S{i_subj}_{activity}.{camera_name}.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 12, (w, h))
                for fr in im_arr:
                    out.write(fr)
                out.release()

                number_activity =  number_activity+1

else:
    data_root = '/home/jovyan/data/'
    if "inaki" in input_path:
        data_root += 'inaki/'
        frame_skip = 2
        frame_rate = 15
    total_frames = len(glob.glob(os.path.join(data_root, input_path, 'frame*.jpg')))
    output_path = (exp_root + "visualize/" + outname)
    prep_dir(output_path)
    prep_dir(output_path+'/'+input_path)
    im_arr = []
    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])
    for i_frame in range(0, total_frames, frame_step):
        save_path = output_path + '/' + input_path + f'/frame_{i_frame}.jpg'
        input_file = os.path.join(data_root, input_path, f'frame{i_frame}.jpg')
        # print(input_file)
        img = Image.open(input_file)
        bbox = model_d.detector.predict_single_image(img)

        x, y, wd, ht, conf = bbox[0]
        if wd < ht:
            y_sq, ht_sq = y, ht
            x_sq, wd_sq = x - (ht - wd) / 2., ht
        else:
            x_sq, wd_sq = x, wd
            y_sq, ht_sq = y - (wd - ht) / 2., wd

        crop_sq = get_crop(img, x_sq, y_sq, wd_sq, ht_sq)
        pred_w_sq = []
        for i_m, model in enumerate(models):
            tt_sq, res_sq = get_pred(crop_sq, input_size[i_m], model, camR)
            pred_w_sq.append(tt_sq)

        fig = plt.figure(figsize=(fsz * nmdl + fsz, fsz * len(views)))
        # Add a 3D subplot

        for i_m in range(nmdl):
            tt_sq = pred_w_sq[i_m].numpy()
            plot_fn = plot_skeleton if len(tt_sq) > 8 else upper_plot_skeleton
            for i_v in range(len(views)):
                ax = fig.add_subplot(len(views), nmdl + 1, (nmdl + 1) * i_v + i_m + 2, projection='3d')
                ax.view_init(*views[i_v])
                plot_fn(ax, tt_sq, 'r')

                # ax.set_xlabel('x')
                ax.set_xlim3d(-700, 700)
                ax.set_zlim3d(-700, 700)
                ax.set_ylim3d(-700, 700)
                if i_v == 0:
                    ax.set_title(f"{model_name[i_m][-3].split('_')[2]}_in{input_size[i_m]}")

        ax2 = fig.add_subplot(len(views), nmdl + 1, 1)
        ax2.imshow(img)
        rect = patches.Rectangle((x, y), wd, ht, linewidth=1, edgecolor='r', facecolor='none')
        rect_sq = patches.Rectangle((x_sq, y_sq), wd_sq, ht_sq, linewidth=1, edgecolor='b', facecolor='none')
        ax2.add_patch(rect_sq)
        ax2.add_patch(rect)
        ax2.set_title(input_path)

        ax3 = fig.add_subplot(len(views), nmdl + 1, nmdl + 2)
        ax3.imshow(res_sq)

        plt.axis('off')
        plt.savefig(save_path)

        plt.close()

        img = cv2.imread(save_path)
        h, w, c = img.shape
        im_arr.append(img)

    out = cv2.VideoWriter(output_path + f'_{input_path}.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), frame_rate/frame_skip, (w, h))
    for fr in im_arr:
        out.write(fr)
    out.release()
