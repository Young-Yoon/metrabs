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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--root-last', action=options.BoolAction)
    options.initialize(parser)
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, f'{paths.DATA_ROOT}/experiments')

    all_image_relpaths, all_true3d = get_all_gt_poses()
    metrics = evaluate(FLAGS.pred_path, all_true3d)
    print(to_latex(metrics))


def evaluate(pred_path, all_true3d):
    all_pred3d = get_all_pred_poses(pred_path)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]

    if all_pred3d.shape[1] == 8:
        dist = np.linalg.norm(all_true3d[:, 9:, :] - all_pred3d, axis=-1)
        return np.mean(dist)
    else:
        dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
        mean_dist_upper = np.mean(np.linalg.norm(all_true3d[:, 9:, :] - all_pred3d[:, 9:, :], axis=-1))
        mean_dist_lower = np.mean(np.linalg.norm(all_true3d[:, :9, :] - all_pred3d[:, :9, :], axis=-1))
        return [np.mean(dist), mean_dist_upper, mean_dist_lower]


def to_latex(numbers):
    return ', '.join([f'{x:.1f}' for x in numbers])


def get_all_gt_poses():
    root_sway = f'{paths.DATA_ROOT}/sway'
    frame_step = 64
    all_world_coords = []
    all_image_relpaths = []

    i_relevant_joints = [2, 5, 8, 1, 4, 7, 9, 12, 15, 15, 16, 18, 20, 17, 19, 21, 0]
    if not FLAGS.root_last:
        i_relevant_joints = [i_relevant_joints[-1]] + i_relevant_joints[:-1]

    with open(f'{root_sway}/test.txt', "r") as f:
        seq_names = [line.strip() for line in f.readlines()]
    for seq_name in util.progressbar(seq_names[:3000]):
        seq_path = os.path.join(root_sway, 'sway61769', seq_name)
        world_pose3d = np.load(os.path.join(seq_path, "wspace_poses3d.npy"))
        if np.isnan(world_pose3d).any():
            continue
        bbox = np.load(os.path.join(seq_path, "bbox.npy"))
        if np.isnan(bbox).any():
            continue

        n_frames_total = len(world_pose3d)
        world_coords = world_pose3d[::frame_step]
        world_coords = world_coords[:, i_relevant_joints, :]
        all_world_coords.append(world_coords)
        image_relfolder = f'sway/sway61769/{seq_name}/images'
        all_image_relpaths += [
            f'{image_relfolder}/{i_frame+1:05d}.jpg'
            for i_frame in range(0, n_frames_total, frame_step)]

    order = np.argsort(all_image_relpaths)
    all_world_coords = np.concatenate(all_world_coords, axis=0)[order]
    all_image_relpaths = np.array(all_image_relpaths)[order]
    return all_image_relpaths, all_world_coords


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    return results['coords3d_pred_world'][order]


if __name__ == '__main__':
    main()
