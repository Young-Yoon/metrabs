#!/usr/bin/env python3
import os

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
    model = tf.saved_model.load(download_model('metrabs_mob3s_y4t'))

    outname = 'metrabs_mob3s'
    exp_root = '/home/jovyan/runs/metrabs-exp/'
    data_root = '/home/jovyan/data/'    
    frame_step = 1
    frame_rate = 15
    
    for test_set in ['inaki', 'kapadia']:
        for subdir in sorted(os.listdir(os.path.join(data_root, test_set))):
            if not os.path.isfile(os.path.join(data_root, test_set, subdir)):
    
                print(subdir)

                data_path = data_root
                input_dir = subdir

                if "inaki" in input_dir:
                    data_path += 'inaki/'
                if "kapadia" in input_dir:
                    data_path += 'kapadia/'


                total_frames = len(glob.glob(os.path.join(data_path, input_dir, 'frame*.jpg')))
                output_path = (exp_root + "visualize/" + outname)
                prep_dir(output_path)
                prep_dir(output_path+'/'+input_dir)
                im_arr = []
                
                for i_frame in tqdm(range(0, total_frames, frame_step)):
                    save_path = output_path + '/' + input_dir + f'/frame_{i_frame}.jpg'
                    input_file = os.path.join(data_path, input_dir, f'frame{i_frame}.jpg')
                    print(input_file)    


                    image = tf.image.decode_jpeg(tf.io.read_file(input_file))
                    #skeleton = 'smpl_24'
                    skeleton = 'h36m_17'

                    pred = model.detect_poses(image, default_fov_degrees=55, skeleton=skeleton)
                    pred = tf.nest.map_structure(lambda x: x.numpy(), pred)  # convert tensors to numpy arrays
#                    print(pred['boxes'], pred['poses3d'], pred['poses2d'])

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

    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
#    plt.switch_backend('TkAgg')

    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    pose_ax.set_box_aspect((1, 1, 1))

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in joint_edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    main()
