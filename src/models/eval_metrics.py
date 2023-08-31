import numpy as np
import tensorflow as tf
from attrdict import AttrDict

import data.datasets3d
import tfu
import tfu3d
from options import FLAGS


def compute_pose3d_metrics(inps, preds):
    metrics = AttrDict()
    coords3d_pred = (
        preds.coords3d_pred_abs if 'coords3d_pred_abs' in preds else preds.coords3d_rel_pred)
    rootrelative_diff = tfu3d.center_relative_pose(
        coords3d_pred - inps.coords3d_true, inps.joint_validity_mask,
        center_is_mean=FLAGS.mean_relative)
    dist = tf.norm(rootrelative_diff, axis=-1)
    metrics.mean_error = tfu.reduce_mean_masked(dist, inps.joint_validity_mask)
    if 'coords3d_pred_abs' in preds:
        metrics.mean_error_abs = tfu.reduce_mean_masked(
            tf.norm(inps.coords3d_true - preds.coords3d_pred_abs, axis=-1),
            inps.joint_validity_mask)

    if 'coords2d_pred' in preds:
        metrics.mean_error_2d = tfu.reduce_mean_masked(
            tf.norm(inps.coords2d_true - preds.coords2d_pred[:, :, :2], axis=-1),
            inps.joint_validity_mask)

    coords3d_pred_procrustes = tfu3d.rigid_align(
        coords3d_pred, inps.coords3d_true,
        joint_validity_mask=inps.joint_validity_mask, scale_align=True)

    dist_procrustes = tf.norm(coords3d_pred_procrustes - inps.coords3d_true, axis=-1)
    metrics.mean_error_procrustes = tfu.reduce_mean_masked(
        dist_procrustes, inps.joint_validity_mask)

    j = data.datasets3d.get_dataset(FLAGS.dataset).joint_info.ids
    threshold = np.float32(150)

    auc_score = tfu.auc(dist, 0, threshold)
    metrics.mean_auc = tfu.reduce_mean_masked(auc_score, inps.joint_validity_mask)

    is_correct = tf.cast(dist <= threshold, tf.float32)
    metrics.mean_pck = tfu.reduce_mean_masked(is_correct, inps.joint_validity_mask)

    if 'lwri' in j and 'rwri' in j:
        all_wrists = [idx for name, idx in j.items() if 'lwri' in name or 'rwri' in name]
        metrics.pck_wrists = tfu.reduce_mean_masked(
            tf.gather(is_correct, all_wrists, axis=1),
            tf.gather(inps.joint_validity_mask, all_wrists, axis=1))
        metrics.auc_wrists = tfu.reduce_mean_masked(
            tf.gather(auc_score, all_wrists, axis=1),
            tf.gather(inps.joint_validity_mask, all_wrists, axis=1))
    else:
        metrics.auc_wrists = tf.constant(0)
        metrics.pck_wrists = tf.constant(0)

    masked_dist_pa = tf.where(inps.joint_validity_mask, dist_procrustes, tf.cast(0, tf.float32))
    max_dist_pa = tf.reduce_max(masked_dist_pa, axis=1)
    metrics.ncps_auc = tf.reduce_mean(tfu.auc(max_dist_pa, 50, 150))
    metrics.ncps = tf.reduce_mean(tf.cast(max_dist_pa <= threshold, tf.float32))
    return metrics






def compute_pose3d_metrics_j8(inps, preds):
    metrics = AttrDict()
    coords3d_pred = (
        preds.coords3d_pred_abs if 'coords3d_pred_abs' in preds else preds.coords3d_rel_pred)
    rootrelative_diff = tfu3d.center_relative_pose(
        coords3d_pred - inps.coords3d_true[:,9:,:], inps.joint_validity_mask[:,9:],
        center_is_mean=FLAGS.mean_relative)
    dist = tf.norm(rootrelative_diff, axis=-1)
    metrics.mean_error = tfu.reduce_mean_masked(dist, inps.joint_validity_mask[:,9:])
    if 'coords3d_pred_abs' in preds:
        metrics.mean_error_abs = tfu.reduce_mean_masked(
            tf.norm(inps.coords3d_true[:,9:,:] - preds.coords3d_pred_abs, axis=-1),
            inps.joint_validity_mask[:,9:])

    if 'coords2d_pred' in preds:
        metrics.mean_error_2d = tfu.reduce_mean_masked(
            tf.norm(inps.coords2d_true[:,9:,:] - preds.coords2d_pred[:, :, :2], axis=-1),
            inps.joint_validity_mask[:,9:])

    coords3d_pred_procrustes = tfu3d.rigid_align(
        coords3d_pred, inps.coords3d_true[:,9:,:],
        joint_validity_mask=inps.joint_validity_mask[:,9:], scale_align=True)

    dist_procrustes = tf.norm(coords3d_pred_procrustes - inps.coords3d_true[:,9:,:], axis=-1)
    metrics.mean_error_procrustes = tfu.reduce_mean_masked(
        dist_procrustes, inps.joint_validity_mask[:,9:])

    j = data.datasets3d.get_dataset(FLAGS.dataset).joint_info.ids
    threshold = np.float32(150)

    auc_score = tfu.auc(dist, 0, threshold)
    metrics.mean_auc = tfu.reduce_mean_masked(auc_score, inps.joint_validity_mask[:,9:])

    is_correct = tf.cast(dist <= threshold, tf.float32)
    metrics.mean_pck = tfu.reduce_mean_masked(is_correct, inps.joint_validity_mask[:,9:])

    if 'lwri' in j and 'rwri' in j:
        all_wrists = [idx for name, idx in j.items() if 'lwri' in name or 'rwri' in name]
        metrics.pck_wrists = tfu.reduce_mean_masked(
            tf.gather(is_correct, all_wrists, axis=1),
            tf.gather(inps.joint_validity_mask[:,9:], all_wrists, axis=1))
        metrics.auc_wrists = tfu.reduce_mean_masked(
            tf.gather(auc_score, all_wrists, axis=1),
            tf.gather(inps.joint_validity_mask[:,9:], all_wrists, axis=1))
    else:
        metrics.auc_wrists = tf.constant(0)
        metrics.pck_wrists = tf.constant(0)

    masked_dist_pa = tf.where(inps.joint_validity_mask[:,9:], dist_procrustes, tf.cast(0, tf.float32))
    max_dist_pa = tf.reduce_max(masked_dist_pa, axis=1)
    metrics.ncps_auc = tf.reduce_mean(tfu.auc(max_dist_pa, 50, 150))
    metrics.ncps = tf.reduce_mean(tf.cast(max_dist_pa <= threshold, tf.float32))
    return metrics

## Metrics calculation for METRABS when using simcc head with softmax 
## Utilized in metrabs_simcc_soft_max.py: MetrabsSimCCSoftMaxTrainer
def compute_pose3d_metrics_simcc_soft_max(inps, preds):
    metrics = AttrDict()


    prob_output3d_idx = tf.argmax(preds.coords3d_rel_pred, axis=-1)
    prob_output2d_idx = tf.argmax(preds.coords2d_pred, axis=-1)


    prob_output3d_idx = tf.cast(prob_output3d_idx, tf.float32)   
    prob_output2d_idx = tf.cast(prob_output2d_idx, tf.float32)   


    input_coord3d = inps.coords3d_true[:, 9:, :]
    center = input_coord3d[:, -1:]    
    input_coord3d = input_coord3d - center + FLAGS.box_size_mm/2

#     rootrelative_diff = tfu3d.center_relative_pose(
#         prob_output3d_idx - inps.coords3d_true[:,9:,:], inps.joint_validity_mask[:,9:],
#         center_is_mean=FLAGS.mean_relative)

    rootrelative_diff = tfu3d.center_relative_pose(
        prob_output3d_idx - input_coord3d, inps.joint_validity_mask[:,9:],
        center_is_mean=FLAGS.mean_relative)


    dist = tf.norm(rootrelative_diff, axis=-1)
    metrics.mean_error = tfu.reduce_mean_masked(dist, inps.joint_validity_mask[:,9:])


    metrics.mean_error_2d = tfu.reduce_mean_masked(
        tf.norm(inps.coords2d_true[:,9:,:] -prob_output2d_idx[:, :, :2], axis=-1),
        inps.joint_validity_mask[:,9:])



    return metrics

## Metrics calculation for METRABS when using simcc head with argmax 
## Utilized in metrabs_simcc_soft_argmax.py: MetrabsSimCCSoftArgMaxTrainer
def compute_pose3d_metrics_simcc_soft_argmax(inps, preds):
    metrics = AttrDict()

    input_coord3d = inps.coords3d_true[:, 9:, :]
    center = input_coord3d[:, -1:]    
    input_coord3d = input_coord3d - center + FLAGS.box_size_mm/2

#     rootrelative_diff = tfu3d.center_relative_pose(
#         prob_output3d_idx - inps.coords3d_true[:,9:,:], inps.joint_validity_mask[:,9:],
#         center_is_mean=FLAGS.mean_relative)

    rootrelative_diff = tfu3d.center_relative_pose(
        preds.coords3d_rel_pred - input_coord3d, inps.joint_validity_mask[:,9:],
        center_is_mean=FLAGS.mean_relative)


    dist = tf.norm(rootrelative_diff, axis=-1)
    metrics.mean_error = tfu.reduce_mean_masked(dist, inps.joint_validity_mask[:,9:])


    metrics.mean_error_2d = tfu.reduce_mean_masked(
        tf.norm(inps.coords2d_true[:,9:,:] -preds.coords2d_pred[:, :, :2], axis=-1),
        inps.joint_validity_mask[:,9:])



    return metrics