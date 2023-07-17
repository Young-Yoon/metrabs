#!/usr/bin/env python3
import os
import itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import itertools
import re

import numpy as np
import spacepy

import data.sway
import options
import paths
import util
from options import FLAGS
import tfu3d
import cameralib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=options.BoolAction)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--root-last', action=options.BoolAction)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    all_image_relpaths, all_true3d = get_all_gt_poses()
    get_next = lambda x: '' if x is None else x[1]
    test_var = np.array([get_next(re.search(f'test_variants/(.+?)\/', path))
                         for path in all_image_relpaths])
    metrics = evaluate(FLAGS.pred_path, all_true3d, test_var)
    print(to_latex(metrics))


def evaluate(pred_path, all_true3d, test_var):
    all_pred3d = get_all_pred_poses(pred_path)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]
    
    test_variants = ('')  # ('', 'landscape', 'portrait', 'tight')
    n_joints_up = 8
    n_joints_pred = all_pred3d.shape[1]
    if FLAGS.procrustes:
        print(all_pred3d.dtype, all_true3d.dtype)
        all_pred3d = tfu3d.rigid_align(all_pred3d, all_true3d[:, -n_joints_pred:], scale_align=True)
    if n_joints_pred == n_joints_up:
        dist = np.linalg.norm(all_true3d[:, -n_joints_pred:] - all_pred3d, axis=-1)
        metrics = [np.mean(dist)]
    else:
        dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
        mean_dist_upper = np.mean(np.linalg.norm(all_true3d[:, -n_joints_up:] - all_pred3d[:, -n_joints_up:], axis=-1))
        mean_dist_lower = np.mean(np.linalg.norm(all_true3d[:, :-n_joints_up] - all_pred3d[:, :-n_joints_up], axis=-1))
        metrics = [np.mean(dist), mean_dist_upper, mean_dist_lower]
    metrics += [np.mean(dist[test_var == tv]) for tv in test_variants]
    return metrics


def to_latex(numbers):
    return ', '.join([f'{x:.1f}' for x in numbers])


@util.cache_result_on_disk(f'{paths.CACHE_DIR}/swaykd_gt_test.pkl', min_time="2023-06-27T11:30:43")
def get_all_gt_poses(use_kd=True):
    all_world_coords = []
    all_image_relpaths = []
    root_sway = f'{paths.DATA_ROOT}/sway'
    seq_names, seq_folders, frame_step = data.sway.get_seq_info('test', root_sway, use_kd)
    print('frame_step', frame_step)

    if use_kd:
        i_relevant_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0]
    else:
        i_relevant_joints = [2, 5, 8, 1, 4, 7, 9, 12, 15, 15, 16, 18, 20, 17, 19, 21, 0]
    if not FLAGS.root_last:
        i_relevant_joints = [i_relevant_joints[-1]] + i_relevant_joints[:-1]

    for seq_dir, seq_name in util.progressbar(itertools.product(seq_folders, seq_names)):
        load_success, params = data.sway.load_seq_param(seq_dir, seq_name, root_sway, use_kd)
        if not load_success:
            continue
        camera, world_pose3d, _, n_frames = params
        if isinstance(camera, cameralib.Camera):
            fixedCam = True
        else:
            intrinsics, _ = camera
            fixedCam = False

        image_relfolder = f'sway/{seq_dir}/{seq_name}/images'
        if use_kd:
            fr_idx = [f'{i_frame+1:05d}' for i_frame in range(0, n_frames, frame_step) 
                      if f'{i_frame+1:05d}' in world_pose3d.keys() and (fixedCam or f'{i_frame+1:05d}' in intrinsics.keys())]
            world_coords = np.array([world_pose3d[fr][i_relevant_joints] for fr in fr_idx], dtype=np.single)
            all_image_relpaths += [f'{image_relfolder}/{fr}.jpg' for fr in fr_idx]
        else:
            world_coords = world_pose3d[::frame_step]
            world_coords = world_coords[:, i_relevant_joints].astype(np.single)
            all_image_relpaths += [
                f'{image_relfolder}/{i_frame+1:05d}.jpg'
                for i_frame in range(0, n_frames, frame_step)]
        all_world_coords.append(world_coords)

    order = np.argsort(all_image_relpaths)
    print(all_world_coords)
    all_world_coords = np.concatenate(all_world_coords, axis=0)[order]
    all_image_relpaths = np.array(all_image_relpaths)[order]
    return all_image_relpaths, all_world_coords


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    return results['coords3d_pred_world'][order]


if __name__ == '__main__':
    main()
