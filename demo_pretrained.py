#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2

def prep_dir(target_path):
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
        print("created folder : ", target_path)
    else:
        print(target_path, "folder already exists.")


def main():
    model = tf.saved_model.load(download_model('metrabs_eff2l_y4'))
#     model = tf.saved_model.load(download_model('metrabs_mob3s_y4'))

    outname = 'metrabs_eff2l_y4'
    exp_root = '/home/jovyan/runs/metrabs-exp/'
    data_root = '/home/jovyan/data/'    
    frame_step = 1
    frame_rate = 30

    for test_set in ['sway4d004', 'sway4d004landscape', 'sway4d004portrait', 'sway4d004tight','inaki', 'kapadia']:
#     for test_set in ['kapadia']:
        for subdir in sorted(os.listdir(os.path.join(data_root, test_set))):
            if not os.path.isfile(os.path.join(data_root, test_set, subdir)):
    
                data_path = data_root
                input_dir = subdir

                if "inaki" in input_dir:
                    data_path += 'inaki/'
                    frame_step = 1                                        
                elif "kapadia" in input_dir:
                    data_path += 'kapadia/'
                    frame_step = 2                    
                elif "sway" in test_set:
                    #data_path += '/images'
                    input_dir = test_set
                    input_dir += '/images'
                    
                    frame_step = 1  
                else:
                    continue

                frames = sorted(glob.glob(os.path.join(data_path, input_dir, '*.jpg')))
                frames = [f.split('/')[-1] for f in frames]
                total_frames = len(frames)
                if total_frames == 0:
                    print(f'No frames in the folder {os.path.join(data_path, input_dir)}')
                
                
                output_path = (exp_root + "visualize/" + outname)
                prep_dir(output_path)
                prep_dir(output_path + '/' + input_dir)
                im_arr = []
                
                for i_fr, frame in enumerate(tqdm(frames[::frame_step])):
                    save_path = output_path + '/' + input_dir + '/' + frame
                    input_file = os.path.join(data_path, input_dir, frame)

                    image = tf.image.decode_jpeg(tf.io.read_file(input_file))
                    #skeleton = 'smpl_24'
                    skeleton = 'h36m_17'
                    
                    if i_fr == 0:
                        pred = model.detect_poses(image, default_fov_degrees=55, skeleton=skeleton)
                    else:
                        keypoints = pred['poses2d'][0]
                        xmin, ymin, xmax, ymax = 99999, 99999, -1, -1
                        for kpt in keypoints:
                            xmin = min(xmin, kpt[0])
                            ymin = min(ymin, kpt[1])
                            xmax = max(xmax, kpt[0])
                            ymax = max(ymax, kpt[1])
                        xmin = max(xmin - 20, 0)
                        ymin = max(ymin - 80, 0)
                        xmax = min(xmax + 20, image.shape[1])
                        ymax = min(ymax + 20, image.shape[0])
                        bbox = tf.convert_to_tensor(np.array([[xmin, ymin, xmax - xmin, ymax - ymin]], dtype=np.float32))
                        pred = model.estimate_poses(image, bbox, default_fov_degrees=55, skeleton=skeleton)
                        pred['boxes'] = bbox

                    pred = tf.nest.map_structure(lambda x: x.numpy(), pred)  # convert tensors to numpy arrays

                    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
                    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

                    visualize(image.numpy(), pred, joint_names, joint_edges, save_path)
                    img = cv2.imread(save_path)
                    h, w, c = img.shape
                    im_arr.append(img)
                    
                out = cv2.VideoWriter(output_path + f'/{input_dir}.mp4',
                                      cv2.VideoWriter_fourcc(*'mp4v'), frame_rate/frame_step, (w, h))
                for fr in im_arr:
                    out.write(fr)
                out.release()
#                 exit()
            
    # Read the docs to learn how to
    # - supply your own bounding boxes
    # - perform multi-image (batched) prediction
    # - supply the intrinsic and extrinsic camera calibration matrices
    # - select the skeleton convention (COCO, H36M, SMPL...)
    # etc.


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


def visualize(image, pred, joint_names, joint_edges, save_path):
#     try:
#         visualize_poseviz(image, pred, joint_names, joint_edges)
#     except ImportError:
#         print(
#             'Install PoseViz from https://github.com/isarandi/poseviz to get a nicer 3D'
#             'visualization.')
    visualize_matplotlib(image, pred, joint_names, joint_edges, save_path)


def visualize_poseviz(image, pred, joint_names, joint_edges):
    # Install PoseViz from https://github.com/isarandi/poseviz
    import poseviz
    camera = poseviz.Camera.from_fov(55, image.shape)
    viz = poseviz.PoseViz(joint_names, joint_edges)
    viz.update(frame=image, boxes=pred['boxes'], poses=pred['poses3d'], camera=camera)


def visualize_matplotlib(image, pred, joint_names, joint_edges, save_path):
    detections, poses3d, poses2d = pred['boxes'], pred['poses3d'], pred['poses2d']
    deg = 5
    views = [(deg, deg - 90), (deg, deg), (90 - deg, deg - 90)]

    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
#    plt.switch_backend('TkAgg')
    upper_indices = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    fig = plt.figure(figsize=(8, 10))
    image_ax = fig.add_subplot(2, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for jj, (i_start, i_end) in enumerate(joint_edges):
            if i_start in upper_indices and i_end in upper_indices:
                image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d[upper_indices].T, s=2)
        
    for vv, view in enumerate(views):
        pose_ax = fig.add_subplot(3, 2, (vv + 1) * 2, projection='3d')
        pose_ax.view_init(*view)
        pose_ax.set_xlim3d(-1000, 1000)
        pose_ax.set_zlim3d(-1000, 1000)
        pose_ax.set_ylim3d(800, 3000)
        pose_ax.set_box_aspect((1, 1, 1))

        for pose3d, pose2d in zip(poses3d, poses2d):
            for jj, (i_start, i_end) in enumerate(joint_edges):
                pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
            pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.clf()


if __name__ == '__main__':
    main()
