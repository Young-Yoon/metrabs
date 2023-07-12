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
from tqdm import tqdm
import argparse


def load_coords(path):
    i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    with spacepy.pycdf.CDF(path) as cdf_file:
        coords_raw = np.array(cdf_file['Pose'], np.float32)[0]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    return coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]


def load_coords_sway(path):
    i_relevant_joints = [2, 5, 8, 1, 4, 7, 9, 12, 15, 15, 16, 18, 20, 17, 19, 21, 0]
    world_pose3d = np.load(path)
    world_coords = world_pose3d[:, i_relevant_joints]
    world_coords -= world_coords[:, -1, np.newaxis]
    return world_coords


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
        

def adjust_bbox(x0, y0, w0, h0, upbb=False, mode=0): 
    if upbb:
        h0 *= 0.5
    
    if mode in [0, 1, 2]:    
        if w0 < h0:
            w1, h1 = h0, h0
            x1, y1 = x0 - (h0 - w0) / 2., y0
        else:
            w1, h1 = w0, w0  
            x1, y1 = x0, y0 - (w0 - h0) / 2.
        return x1, y1, w1, h1
    elif mode == 3:
        if w0 < h0:
            w1, h1 = w0, w0
            x1, y1 = x0, y0 + (h0 - w0) * 0.1
        else:
            w1, h1 = h0, h0  
            x1, y1 = x0 + (w0 - h0) * 0.5, y0
        return x1, y1, w1, h1
    else:
        print("Error. Unsupported bbox squarization mode. Only 0, 1, 2, 3 are supported!")
        exit()


def get_crop(img0, x0, y0, w0, h0, zeropadding=False):
    x1 = x0 + w0
    y1 = y0 + h0
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, img0.width)
    y1 = min(y1, img0.height)
    crop_img = np.array(img0)[int(y0):int(y1), int(x0):int(x1)]
    if zeropadding:
        crop_h, crop_w, _ = crop_img.shape
        crop_l = max(crop_h, crop_w)
        zp_crop_img = np.zeros([crop_l, crop_l, 3], dtype=crop_img.dtype)
        h_offset = int((crop_l - crop_h) / 2)
        w_offset = int((crop_l - crop_w) / 2)
        zp_crop_img[h_offset: h_offset + crop_h, w_offset: w_offset + crop_w] = crop_img
        return zp_crop_img
    else:
        return crop_img

    
def adjust_bbox_and_get_crop(img0, x0, y0, w0, h0, upbb=False, mode=0): 
    if upbb:
        h0 *= 0.5
    if w0 < h0:
        w1, h1 = h0, h0
        x1, y1 = x0 - (h0 - w0) / 2., y0
    else:
        w1, h1 = w0, w0  
        x1, y1 = x0, y0 - (w0 - h0) / 2.    
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x1_max = min(x1_max, img0.width)
    y1_max = min(y1_max, img0.height)  
    w1 = x1_max - x1
    h1 = y1_max - y1
    
    if mode in [0, 1]:    
        crop_img = np.array(img0)[int(y1):int(y1_max), int(x1):int(x1_max)]
        return crop_img, (x1, y1, w1, h1)
    elif mode == 2:
        crop_img = np.array(img0)[int(y1):int(y1_max), int(x1):int(x1_max)]
        crop_h, crop_w, _ = crop_img.shape
        crop_l = max(crop_h, crop_w)
        zp_crop_img = np.zeros([crop_l, crop_l, 3], dtype=crop_img.dtype)
        h_offset = int((crop_l - crop_h) / 2)
        w_offset = int((crop_l - crop_w) / 2)
        zp_crop_img[h_offset: h_offset + crop_h, w_offset: w_offset + crop_w] = crop_img
        return zp_crop_img, (x1, y1, w1, h1)
    elif mode == 3:
        if w1 < h1:
            w3, h3 = w1, w1
            x3, y3 = x1, y1 + (h1 - w1) * 0.1
        else:
            w3, h3 = h1, h1  
            x3, y3 = x1 + (w1 - h1) * 0.5, y1
        x3_max = x3 + w3
        y3_max = y3 + h3
        crop_img = np.array(img0)[int(y3):int(y3_max), int(x3):int(x3_max)]
        return crop_img, (x3, y3, w3, h3)
    else:
        print("Error. Unsupported bbox squarization mode. Only 0, 1, 2, 3 are supported!")
        exit()


def get_crop(img0, x0, y0, w0, h0, zeropadding=False):

    if zeropadding:
        crop_h, crop_w, _ = crop_img.shape
        crop_l = max(crop_h, crop_w)
        zp_crop_img = np.zeros([crop_l, crop_l, 3], dtype=crop_img.dtype)
        h_offset = int((crop_l - crop_h) / 2)
        w_offset = int((crop_l - crop_w) / 2)
        zp_crop_img[h_offset: h_offset + crop_h, w_offset: w_offset + crop_w] = crop_img
        return zp_crop_img
    else:
        return crop_img    


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
#     or python viz.py model_path

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models', nargs='+', help="models for visualization, separated by space")
parser.add_argument('-sq', "--squarization", type=int, default=0, help="method to make a square bbox from a rectangle bbox")
parser.add_argument('-o', "--outname", type=str, help="output folder name")
parser.add_argument('-i', "--input_path", type=str, default="all", help="input video name")
parser.add_argument('-d', "--detector", action='store_true', default=True, help="display a square of a given number")
parser.add_argument('-up', "--upperbbox", action='store_true', help="whether or not to use 0.5 height of fullbody box")
args = parser.parse_args()
print(args)

use_detector = args.detector
model_folders = args.models
input_path = args.input_path
outname = args.outname
upbbox = args.upperbbox
bbox_square_mode = args.squarization
zeropadding = bbox_square_mode == 2

exp_root = '/home/jovyan/runs/metrabs-exp/'
data_root = '/home/jovyan/data/'

#model_folder = exp_root + 'tconv1_in112/h36m_METRO_m3s03_rand1_1500/model/'
model_folder = [exp_root + p + '/model/' for p in model_folders]
for p in model_folder:
    if not os.path.isdir(p):
        raise OSError("model_folder not exist" + p)

model_name = [p.split("/") for p in model_folder]
input_size = [112 if "112" in p else 160 if "160" in p else 128 if "128" in p else 256 for p in model_folders]
print(model_name, input_size)

model_detector_folder = exp_root + "models/eff2s_y4/model_multi"
model_d = tf.saved_model.load(model_detector_folder)
det = model_d.detector

models = [tf.saved_model.load(p) for p in model_folder]
nmdl = len(models)
print("Loading model is done")

deg = 5
views = [(deg, deg - 90), (deg, deg), (90 - deg, deg - 90)]
fsz = 2

def plot_h36m(act_key=None, frame_step=25, data_path=data_root+'metrabs-processed/h36m/'):
    cam_param = exp_root + "visualize/human36m-camera-parameters/camera-parameters.json"
    with open(cam_param, 'r') as j:
         cam = json.loads(j.read())

    camera_names = ['55011271', '54138969', '58860488', '60457274']
    all_world_coords = []
    all_image_relpaths = []

    for i_subj in [11]: # [9, 11]:
        number_activity=0
        for activity, cam_id in itertools.product(data.h36m.get_activity_names(i_subj), range(4)):
                
                if number_activity==4:
                    break
                if cam_id > 0:
                    continue
                
                print(activity, cam_id, number_activity)
                im_arr = []

                if act_key is not None and not act_key in activity:
                    continue
                cam_p = cam['extrinsics']['S'+str(i_subj)][camera_names[cam_id]]
                cam_rot = np.expand_dims(np.linalg.inv(np.array(cam_p['R'])), axis=0)  # (1,3,3)
                cam_loc = np.expand_dims(np.array(cam_p['t'])[:,0], axis=0)            # (1,3)

                camera_name = camera_names[cam_id]
                pose_folder = data_path + f'S{i_subj}/MyPoseFeatures'
                coord_path = f'{pose_folder}/D3_Positions/{activity}.cdf'
                world_coords = load_coords(coord_path)
                world_coords -= world_coords[:, -1, np.newaxis]

                n_frames_total = len(world_coords)
                image_relfolder = data_path + f'S{i_subj}/Images/{activity}.{camera_name}'

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

                total_frame = n_frames_total//frame_step

                out_test = np.zeros((total_frame,17,3))
                k=0
                bboxes_gt = np.load(f'{data_path}/S{i_subj}/BBoxes/{activity}.{camera_name}.npy')

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
                    if len(bbox) == 0:
                        x, y, wd, ht = 0, 0, img.width, img.height
                    else:
                        x, y, wd, ht, conf = bbox[0]
                    x_sq, y_sq, wd_sq, ht_sq = adjust_bbox(x, y, wd, ht, upbbox)
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

                out = cv2.VideoWriter(exp_root + "visualize/" + outname + f'/S{i_subj}_{activity}.{camera_name}.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 12, (w, h))
                for fr in im_arr:
                    out.write(fr)
                out.release()

                number_activity =  number_activity+1


def plot_wild(input_dir, data_path=data_root, frame_step=2, frame_rate=30):
    gt_path = ''
    if "inaki" in input_dir:
        data_path += 'inaki/'
    if "kapadia" in input_dir:
        data_path += 'kapadia/'
        frame_step = 5
    if "sway" in input_dir:
        frame_step = 1
        gt_path = os.path.join(data_path, input_dir, "wspace_poses3d.npy")
        world_coords = load_coords_sway(gt_path)[::frame_step]
        input_dir += '/images'
        
    frames = sorted(glob.glob(os.path.join(data_path, input_dir, '*.jpg')))
    frames = [f.split('/')[-1] for f in frames]
    total_frames = len(frames)
    if total_frames == 0:
        print(f'No frames in the folder {os.path.join(data_path, input_dir)}')
    output_path = (exp_root + "visualize/" + outname)
    prep_dir(output_path)
    prep_dir(output_path+'/'+input_dir)
    im_arr = []
    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])
    npyfiles = np.zeros([nmdl, len(frames[::frame_step]), 8, 3])
    
    for i_fr, frame in enumerate(tqdm(frames[::frame_step])):
        save_path = output_path + '/' + input_dir + '/' + frame
        #print(save_path, input_dir, frame)
        input_file = os.path.join(data_path, input_dir, frame)
        img = Image.open(input_file)
        if use_detector:
            bbox = model_d.detector.predict_single_image(img)
            if len(bbox)==0:
                continue
            x, y, wd, ht, conf = bbox[0]
            # x_sq, y_sq, wd_sq, ht_sq = adjust_bbox(x, y, wd, ht, upbbox, mode=bbox_square_mode)
            # crop_sq = get_crop(img, x_sq, y_sq, wd_sq, ht_sq, zeropadding=zeropadding)
            crop_sq, crop_bbox = adjust_bbox_and_get_crop(img, x, y, wd, ht, upbb=upbbox, mode=bbox_square_mode)
            x_sq, y_sq, wd_sq, ht_sq = crop_bbox

        else:
            crop_sq = np.array(img)
        pred_w_sq = []
        for i_m, model in enumerate(models):
            tt_sq, res_sq = get_pred(crop_sq, input_size[i_m], model, camR)
            npyfiles[i_m, i_fr] = tt_sq.numpy()
            pred_w_sq.append(tt_sq)

        fig = plt.figure(figsize=(fsz * nmdl + fsz, fsz * len(views)))
        # Add a 3D subplot

        for i_m in range(nmdl):
            tt_sq = pred_w_sq[i_m].numpy()
            plot_fn = plot_skeleton if len(tt_sq) > 8 else upper_plot_skeleton
            for i_v in range(len(views)):
                ax = fig.add_subplot(len(views), nmdl + 1, (nmdl + 1) * i_v + i_m + 2, projection='3d')
                ax.view_init(*views[i_v])

                if bool(gt_path):
                    gt = world_coords[i_fr]
                    plot_skeleton(ax, gt, 'b')
                plot_fn(ax, tt_sq, 'r')

                # ax.set_xlabel('x')
                ax.set_xlim3d(-700, 700)
                ax.set_zlim3d(-700, 700)
                ax.set_ylim3d(-700, 700)
                if i_v == 0:
                    mdl_name = model_folders[i_m]
                    if len(mdl_name)>30:
                        mdl_name = mdl_name.split('/')[-1]
                        if len(mdl_name) > 30:
                            mdl_name = mdl_name[:30]
                    ax.set_title(f"{mdl_name}")
                    ax.title.set_size(180/len(mdl_name))

        ax2 = fig.add_subplot(len(views), nmdl + 1, 1)
        ax2.imshow(img)
        rect = patches.Rectangle((x, y), wd, ht, linewidth=1, edgecolor='r', facecolor='none')
        rect_sq = patches.Rectangle((x_sq, y_sq), wd_sq, ht_sq, linewidth=1, edgecolor='b', facecolor='none')
        ax2.add_patch(rect_sq)
        ax2.add_patch(rect)
        ax2.set_title(input_dir)
        ax2.title.set_size(np.min([130/len(input_dir), 10]))

        ax3 = fig.add_subplot(len(views), nmdl + 1, nmdl + 2)
        ax3.imshow(res_sq)

        plt.axis('off')
        plt.savefig(save_path)

        plt.close()

        img = cv2.imread(save_path)
        h, w, c = img.shape
        im_arr.append(img)

    out = cv2.VideoWriter(output_path + f'/{input_dir.replace("/","_")}.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), frame_rate/frame_step, (w, h))
    for fr in im_arr:
        out.write(fr)
    out.release()

    bbox_name = "upperbbox" if upbbox else "fullbbox"
    for i_m in range(nmdl):
        mdl_name = model_folders[i_m]
        np.save(output_path + f'/{os.path.split(mdl_name)[1]}_{input_dir.replace("/images", "")}_{bbox_name}_sq{bbox_square_mode}.npy', npyfiles[i_m])
    return


if input_path in {'all', 'wild'}:
    plot_wild('sway4d004')
    for test_set in ['inaki', 'kapadia']:
        for subdir in sorted(os.listdir(os.path.join(data_root, test_set))):
            if subdir.startswith(test_set) and not os.path.isfile(os.path.join(data_root, test_set, subdir)):
                print("Processing ", data_root, test_set, subdir)
                plot_wild(subdir)
    if input_path in {'all'}:
        plot_h36m()
elif input_path.startswith('h36m'):
    plot_h36m(input_path[4:])
else:
    plot_wild(input_path)
