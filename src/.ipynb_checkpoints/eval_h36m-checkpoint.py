#!/usr/bin/env python3
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import itertools
import re

import numpy as np
import spacepy

import data.h36m
import options
import paths
import util
from options import FLAGS
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
#plt.switch_backend('TkAgg')

import tfu3d

def main():
#    FLAGS.only_S11 =True

    FLAGS.root_last= True

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    
    parser.add_argument('--procrustes', action=options.BoolAction)
    parser.add_argument('--only-S11', action=options.BoolAction)
    parser.add_argument('--seeds', type=int, default=1)

    # The root joint is the last if this is set, else the first
    parser.add_argument('--root-last', action=options.BoolAction)
    options.initialize(parser)
#    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

#    print("pred path : ", FLAGS.pred_path)

    all_image_relpaths, all_true3d = get_all_gt_poses()
#    print(len(all_true3d))

    activities = np.array([re.search(f'Images/(.+?)\.', path)[1].split(' ')[0]
                           for path in all_image_relpaths])

    if FLAGS.seeds > 1:
        mean_per_seed, std_per_seed = evaluate_multiple_seeds(all_true3d, activities)
        print(to_latex(mean_per_seed))
        print(to_latex(std_per_seed))
    else:
        metrics = evaluate(FLAGS.pred_path, all_true3d, activities, all_image_relpaths )
        print(to_latex(metrics))


def evaluate_multiple_seeds(all_true3d, activities):
    seed_pred_paths = [FLAGS.pred_path.replace('seed1', f'seed{i + 1}') for i in range(FLAGS.seeds)]
    metrics_per_seed = np.array([evaluate(p, all_true3d, activities) for p in seed_pred_paths])
    mean_per_seed = np.mean(metrics_per_seed, axis=0)
    std_per_seed = np.std(metrics_per_seed, axis=0)
    return mean_per_seed, std_per_seed


skeleton_gt = [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12), 
            (7, 13), (13, 14), (14, 15), (7, 6), (6, 16), 
            (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]  # head

skeleton = [(1, 2), (2, 3), (4, 5), (5, 6)]  # head


# model jiont info :  [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12), (7, 13), (13, 14), (14, 15), (7, 6), (6, 16), (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]

from PIL import Image



def evaluate(pred_path, all_true3d, activities, all_image_relpaths):
#    print("lets get prediction pose ")
    all_pred3d = get_all_pred_poses(pred_path)
#    print("all prediction pose : ", len(all_pred3d))
#    print(all_pred3d)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
#    print("FLAGS.root_last ", FLAGS.root_last)
    
    all_true3d -= all_true3d[:, i_root, np.newaxis]
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    
    MYDIR = (FLAGS.save_path +"/test/")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")    
    
    for k in range(0,20):

        tt = all_pred3d[k]
        tt2 = all_true3d[k]

        # Create a new figure
        fig = plt.figure()
        # Add a 3D subplot
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(tt[:, 0], tt[:, 1], tt[:, 2], s=10, c='g', alpha=1.0 )
        ax.scatter(tt2[:, 0], tt2[:, 1], tt2[:, 2] , s=1, c='b' )
        
        for i, j in skeleton:
#            plt.plot([tt[i, 0], tt[j, 0]], [-tt[i, 2], -tt[j, 2]], [-tt[i, 1], -tt[j, 1]], 'r')
            plt.plot([tt[i, 0], tt[j, 0]], [tt[i, 1], tt[j, 1]], [tt[i, 2], tt[j, 2]], 'r')            
#            plt.plot([tt2[i, 0], tt2[j, 0]], [tt2[i, 1], tt2[j, 1]], [tt2[i, 2], tt2[j, 2]], 'b', alpha=0.5)


        for i, j in skeleton_gt:
#            plt.plot([tt[i, 0], tt[j, 0]], [-tt[i, 2], -tt[j, 2]], [-tt[i, 1], -tt[j, 1]], 'r')
            plt.plot([tt2[i, 0], tt2[j, 0]], [tt2[i, 1], tt2[j, 1]], [tt2[i, 2], tt2[j, 2]], 'b', alpha=0.3)
        
        # Set labels if necessary
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-500, 500)
        ax.set_zlim3d(-1000, 1000)
        ax.set_ylim3d(-500, 500)

        image = Image.open("/home/jovyan/data/metrabs-processed/"+all_image_relpaths[k])
        ax2 = fig.add_subplot(121)
        ax2.imshow(image)

        # Save the figure to a file
        plt.savefig( FLAGS.save_path +"/test/" + str(k)+'.png')
#        plt.clf()

    ordered_activities = (
            'Directions Discussion Eating Greeting Phoning Posing Purchases ' +
            'Sitting SittingDown Smoking Photo Waiting Walking WalkDog WalkTogether').split()

    dist = np.linalg.norm(all_true3d[:,9:,:] - all_pred3d, axis=-1)
    overall_mean_error = np.mean(dist)
    print("overall mean errror : ", overall_mean_error)
    metrics = [np.mean(dist[activities == activity]) for activity in ordered_activities]
    metrics.append(overall_mean_error)
    return metrics


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' for x in numbers])


def load_coords(path):
    if FLAGS.root_last:
        i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    else:
        i_relevant_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    with spacepy.pycdf.CDF(path) as cdf_file:
        coords_raw = np.array(cdf_file['Pose'], np.float32)[0]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    return coords_raw.reshape(coords_new_shape)[:, i_relevant_joints]


def get_all_gt_poses():
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    frame_step = 64
    all_world_coords = []
    all_image_relpaths = []
    for i_subj in [9, 11]:
        
        for activity, cam_id in itertools.product(data.h36m.get_activity_names(i_subj), range(4)):
            
            # Corrupt data in original release:
            # if i_subj == 11 and activity == 'Directions' and cam_id == 0:
            #    continue
            camera_name = camera_names[cam_id]
            pose_folder = f'{paths.DATA_ROOT}/h36m/S{i_subj}/MyPoseFeatures'
            coord_path = f'{pose_folder}/D3_Positions/{activity}.cdf'
            world_coords = load_coords(coord_path)
            n_frames_total = len(world_coords)
            world_coords = world_coords[::frame_step]
            all_world_coords.append(world_coords)
            image_relfolder = f'h36m/S{i_subj}/Images/{activity}.{camera_name}'
                        
            all_image_relpaths += [
                f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                for i_frame in range(0, n_frames_total, frame_step)]
            

    order = np.argsort(all_image_relpaths)
    all_world_coords = np.concatenate(all_world_coords, axis=0)[order]
    all_image_relpaths = np.array(all_image_relpaths)[order]
    if FLAGS.only_S11:
        needed = ['S11' in p for p in all_image_relpaths]
        return all_image_relpaths[needed], all_world_coords[needed]
    return all_image_relpaths, all_world_coords


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    image_paths = results['image_path'][order]
    if FLAGS.only_S11:
        needed = ['S11' in p for p in image_paths]
        return results['coords3d_pred_world'][order][needed]

    return results['coords3d_pred_world'][order]


if __name__ == '__main__':
    main()