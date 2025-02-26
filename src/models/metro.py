import einops
import keras
import keras.layers
import keras.metrics
import numpy as np
import tensorflow as tf
from attrdict import AttrDict

import models.eval_metrics
import models.model_trainer
import models.util
import tfu
import tfu3d
from options import FLAGS


class Metro(keras.Model):
    def __init__(self, backbone, joint_info):
        super().__init__()
        self.backbone = backbone
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)
        if FLAGS.output_upper_joints:
            n_raw_points = 8
        else:
            n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints
        self.heatmap_head = Head3D(n_points=n_raw_points)
        if FLAGS.transform_coords:
            self.recombination_weights = tf.constant(
                np.load(
                    '/globalwork/sarandi/data/skeleton_conversion/'
                    'latent_to_all_32_singlestage.npy'))

        if FLAGS.data_format == 'NCHW':    
            self.input_shape_image =(None, 3, None, None)
        else:
            self.input_shape_image =(None, None, None, 3)

        self.predict_multi.get_concrete_function(
            tf.TensorSpec(shape=self.input_shape_image, dtype=tf.float32 if FLAGS.input_float32 else tf.float16))

    def call(self, image, training=None):
        features = self.backbone(image, training=training)
        coords3d = self.heatmap_head(features, training=training)
        if FLAGS.transform_coords:
            coords3d = self.latent_points_to_joints(coords3d)
        return coords3d

    @tf.function
    def predict_multi(self, image):
        return self.call(image, training=False)

    def latent_points_to_joints(self, points):
        return tfu3d.linear_combine_points(points, self.recombination_weights)


class Head3D(keras.layers.Layer):
    def __init__(self, n_points):
        super().__init__()
        self.n_points = n_points
        self.conv_final = keras.layers.Conv2D(filters=FLAGS.depth * self.n_points, kernel_size=1)

    def call(self, inp, training=None):
        logits = self.conv_final(inp)
        
        if FLAGS.data_format == 'NCHW':        
            current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (j d) h w'
            logits = einops.rearrange(logits, f'{current_format} -> b j d h w', j=self.n_points)
            coords_heatmap = tfu.soft_argmax(tf.cast(logits, tf.float32), axis=[4, 3, 2])
        else:
            current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (d j) h w'
            logits = einops.rearrange(logits, f'{current_format} -> b h w d j', j=self.n_points)
            coords_heatmap = tfu.soft_argmax(tf.cast(logits, tf.float32), axis=[2, 1, 3])
        
        return models.util.heatmap_to_metric(coords_heatmap, training)


class MetroTrainer(models.model_trainer.ModelTrainer):
    def __init__(self, metro_model, joint_info, joint_info2d=None, global_step=None):
        super().__init__(global_step)
        self.global_step = global_step
        self.joint_info = joint_info
        self.joint_info_2d = joint_info2d
        self.model = metro_model

        if FLAGS.data_format == 'NCHW':        
            inp = keras.Input(shape=(3, None, None), dtype=tfu.get_dtype())
        else:
            inp = keras.Input(shape=(None, None,3), dtype=tfu.get_dtype())


        self.model(inp, training=False)

    def forward_test(self, inps):
        return dict(coords3d_rel_pred=self.model(inps['image'], training=False))

    def forward_train(self, inps, training):
        preds = AttrDict()

        if FLAGS.loss2d_factor > 0.0:
            image_both = tf.concat([inps.image, inps.image_2d], axis=0)
            coords3d_pred_both = self.model(image_both, training=training)
            batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]
            preds.coords3d_rel_pred, preds.coords3d_pred_2d = tf.split(
                coords3d_pred_both, batch_sizes, axis=0)
        else:
            preds.coords3d_rel_pred = self.model(inps.image, training=training)

        joint_ids_3d = [
            [self.joint_info.ids[n2] for n2 in self.joint_info.names if n2.startswith(n1)]
            for n1 in self.joint_info_2d.names]
        
        if FLAGS.output_upper_joints:
            joint_ids_3d= [[6], [5], [4], [1], [2], [3]]

        def get_2dlike_joints(coords):
            return tf.stack(
                [tf.reduce_mean(tf.gather(coords, ids, axis=1)[..., :2], axis=1)
                 for ids in joint_ids_3d], axis=1)

        # numbers mean: like 2d dataset joints, 2d batch
        if FLAGS.loss2d_factor > 0.0:
            preds.coords2d_pred_2d = get_2dlike_joints(preds.coords3d_pred_2d[..., :2])

        return preds

    def compute_losses(self, inps, preds):
        losses = AttrDict()

        ####################
        # 3D BATCH
        ####################  index starts from 0        
        # 'htop': 9, 'lsho': 10, 'lelb': 11, 'lwri': 12, 'rsho': 13, 'relb': 14, 'rwri': 15, 'pelv': 16})
        joint_index_start = 9 if FLAGS.output_upper_joints else 0 
        
        coords3d_true_rootrel = tfu3d.center_relative_pose(
            inps.coords3d_true[:, joint_index_start:, :], inps.joint_validity_mask[:, joint_index_start:], FLAGS.mean_relative)
        coords3d_pred_rootrel = tfu3d.center_relative_pose(
            preds.coords3d_rel_pred, inps.joint_validity_mask[:, joint_index_start:], FLAGS.mean_relative)

        rootrel_absdiff = tf.abs((coords3d_true_rootrel - coords3d_pred_rootrel) / 1000)
        losses.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, inps.joint_validity_mask[:, joint_index_start:])
        if FLAGS.loss2d_factor==0.0:
            losses.loss = losses.loss3d
            return losses

        ####################
        # 2D BATCH
        ####################
#        for 2d joints joint_ids_3d = [[2], [1], [0], [3], [4], [5], [15], [14], [13], [10], [11], [12]]    # we only need 15,14,13,10,11,12
        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000

        joint_index_start = 6 if FLAGS.output_upper_joints else 0
        
        preds.coords2d_pred_2d = models.util.align_2d_skeletons(
            preds.coords2d_pred_2d, inps.coords2d_true_2d[:, joint_index_start:, :], inps.joint_validity_mask_2d[:, joint_index_start:])
        losses.loss2d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d[:, joint_index_start:, :] - preds.coords2d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d[:, joint_index_start:])

        losses.loss = losses.loss3d + FLAGS.loss2d_factor * losses.loss2d
        
        return losses

    @tf.function
    def compute_metrics(self, inps, preds):
#        print("when we use it ? ")
        return models.eval_metrics.compute_pose3d_metrics_j8(inps, preds) if FLAGS.output_upper_joints else models.eval_metrics.compute_pose3d_metrics(inps, preds)
 
