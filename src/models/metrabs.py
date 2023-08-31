
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

class Metrabs(keras.Model):
    def __init__(self, backbone, joint_info):
        super().__init__()
        self.backbone = backbone
        self.joint_names = tf.Variable(np.array(joint_info.names), trainable=False)
        self.joint_edges = tf.Variable(np.array(joint_info.stick_figure_edges), trainable=False)
        self.joint_info = joint_info
        if FLAGS.output_upper_joints:
            n_raw_points = 8
        else:
            n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints   

        ''' 
        Selection of METRABS head class based on flags used. 
        Possible values of head_class_string: 
        - MetrabsHeads (default, if FLAGS.metrabs_simcc_head = "")
        - MetrabsSimCCSoftMaxHeads (if FLAGS.metrabs_simcc_head = SimCCSoftMax)
        - MetrabsSimCCSoftArgMaxHeads (if FLAGS.metrabs_simcc_head = SimCCSoftArgMax)
        '''
        head_class_string = 'Metrabs' + FLAGS.metrabs_simcc_head + 'Heads'

        ## importing here to avoid circular imports 
        import models.metrabs_simcc_soft_argmax as metrabs_simcc_soft_argmax
        import models.metrabs_simcc_soft_max as metrabs_simcc_soft_max

        head_class = getattr(models, head_class_string)

        self.heatmap_heads = head_class(n_points=n_raw_points)      
        #self.heatmap_heads = MetrabsHeads(n_points=n_raw_points)
        if FLAGS.transform_coords:
            self.recombination_weights = tf.constant(np.load('32_to_122'))    

        if FLAGS.data_format == 'NCHW':    
            self.input_shape_image =(None, 3, None, None)
        else:
            self.input_shape_image =(None, None, None, 3)

        self.predict_multi.get_concrete_function(
            tf.TensorSpec(shape=self.input_shape_image, dtype=tf.float32 if FLAGS.input_float32 else tf.float16))

    def call(self, inp, training=None):
        image = inp
        features = self.backbone(image, training=training)


        ''' 
        Updates to reflect new function signature of self.heatmap_heads which now only returns feature vectors. 
        Note that this may not be compatible with simcc heads. 
        '''
        volume_3d = self.heatmap_heads(features, training=training)
        return volume_3d

    @tf.function
    def predict_multi(self, image):
        # This function is needed to avoid having to go through Keras' __call__
        # in the exported SavedModel, which causes all kinds of problems.
        return self.call(image, training=False)

    def latent_points_to_joints(self, points):
        return tfu3d.linear_combine_points(points, self.recombination_weights)

class MetrabsHeads(keras.layers.Layer):
    def __init__(self, n_points):
        super().__init__()
        self.n_points = n_points
        self.n_outs = [self.n_points, FLAGS.depth * self.n_points]
        self.conv_final = keras.layers.Conv2D(filters=sum(self.n_outs), kernel_size=1)

    ## Modified implementation of METRABS head does not have 3D soft argmax and only returns feature vectors. 
    def call(self, inp, training=None):
        x = self.conv_final(inp)
        return x


class MetrabsTrainer(models.model_trainer.ModelTrainer):
    def __init__(self, metrabs_model, joint_info, joint_info2d=None, global_step=None):
        super().__init__(global_step)
        self.global_step = global_step
        self.joint_info = joint_info
        self.joint_info_2d = joint_info2d
        self.model = metrabs_model

        ## Estimating self.n_raw_points based on FLAGS.output_upper_joints 
        if FLAGS.output_upper_joints:
            self.n_raw_points = 8
        else:
            self.n_raw_points = 32 if FLAGS.transform_coords else joint_info.n_joints             
        
        if FLAGS.data_format == 'NCHW':        
            inp = keras.Input(shape=(3, None, None), dtype=tfu.get_dtype())
        else:
            inp = keras.Input(shape=(None, None,3), dtype=tfu.get_dtype())

        ## Variable used for feature splitting into 2d and 3D [8, 8*8]
        self.n_outs = [self.n_raw_points, FLAGS.depth * self.n_raw_points]
            
        self.model(inp, training=False)

    def _shared_process_inps(self, inps, training):
        preds = AttrDict()

        if training:
            image_both = tf.concat([inps.image, inps.image_2d], axis=0)
            features = self.model.backbone(image_both, training=training)
        else:
            features = self.model.backbone(inps.image, training=training)

        volume_3d = self.model.heatmap_heads(features, training=training)
        logits2d, logits3d = tf.split(volume_3d, self.n_outs, axis=tfu.channel_axis())

        if FLAGS.data_format == 'NCHW':        
            current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (j d) h w'
            logits3d = einops.rearrange(logits3d, f'{current_format} -> b j d h w', j=self.n_raw_points)
            coords3d = tfu.soft_argmax(tf.cast(logits3d, tf.float32), axis=[4, 3, 2])
        else:
            current_format = 'b h w (d j)' if tfu.get_data_format() == 'NHWC' else 'b (d j) h w'
            logits3d = einops.rearrange(logits3d, f'{current_format} -> b h w d j', j=self.n_raw_points)
            coords3d = tfu.soft_argmax(tf.cast(logits3d, tf.float32), axis=[2, 1, 3])

        preds.coords3d_rel_pred = models.util.heatmap_to_metric(coords3d, training)
        coords2d = tfu.soft_argmax(tf.cast(logits2d, tf.float32), axis=tfu.image_axes()[::-1])
        preds.coords2d_pred = models.util.heatmap_to_image(coords2d, training)
        
        if training:        
            batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]
            preds.coords2d_pred, preds.coords2d_pred_2d = tf.split(
                preds.coords2d_pred, batch_sizes, axis=0)
            preds.coords3d_rel_pred, preds.coords3d_rel_pred_2d = tf.split(
                preds.coords3d_rel_pred, batch_sizes, axis=0)        
        
        preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
            preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        return preds
    
    def forward_train(self, inps, training):

        preds = self._shared_process_inps (inps, training)   

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred_2d = l2j(preds.coords2d_pred_2d)
            preds.coords3d_rel_pred_2d = l2j(preds.coords3d_rel_pred_2d)
            preds.coords2d_pred_latent = preds.coords2d_pred
            preds.coords2d_pred = l2j(preds.coords2d_pred_latent)
            preds.coords3d_rel_pred_latent = preds.coords3d_rel_pred
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred_latent)
            preds.coords3d_pred_abs = l2j(tfu3d.reconstruct_absolute(
                preds.coords2d_pred_latent, preds.coords3d_rel_pred_latent, inps.intrinsics))
        else:
            preds.coords3d_pred_abs = tfu3d.reconstruct_absolute(
                preds.coords2d_pred, preds.coords3d_rel_pred, inps.intrinsics)

        joint_ids_3d = [
            [self.joint_info.ids[n2] for n2 in self.joint_info.names if n2.startswith(n1)]
            for n1 in self.joint_info_2d.names]
        
        if FLAGS.output_upper_joints:
            joint_ids_3d= [[6], [5], [4], [1], [2], [3]]        

        def get_2dlike_joints(coords):
            return tf.stack(
                [tf.reduce_mean(tf.gather(coords, ids, axis=1)[..., :2], axis=1)
                 for ids in joint_ids_3d], axis=1)

        # numbers mean: 3d head, 2d dataset joints, 2d batch
        preds.coords32d_pred_2d = get_2dlike_joints(preds.coords3d_rel_pred_2d)
        preds.coords22d_pred_2d = get_2dlike_joints(preds.coords2d_pred_2d)
        return preds

    def compute_losses(self, inps, preds):
        losses = AttrDict()


        joint_index_start = 9 if FLAGS.output_upper_joints else 0 
        
        
        if FLAGS.scale_agnostic_loss:
            mean_true, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true[:, joint_index_start:, :], inps.joint_validity_mask[:, joint_index_start:, :], items_axis=1, dimensions_axis=2)
            mean_pred, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_rel_pred, inps.joint_validity_mask[:, joint_index_start:, :], items_axis=1, dimensions_axis=2)
            coords3d_pred_rootrel = tf.math.divide_no_nan(
                preds.coords3d_rel_pred - mean_pred, scale_pred) * scale_true
            coords3d_true_rootrel = inps.coords3d_true[:, joint_index_start:, :] - mean_true
        else:           
            coords3d_true_rootrel = tfu3d.center_relative_pose(
                inps.coords3d_true[:, joint_index_start:, :], inps.joint_validity_mask[:, joint_index_start:], FLAGS.mean_relative)
            coords3d_pred_rootrel = tfu3d.center_relative_pose(
                preds.coords3d_rel_pred, inps.joint_validity_mask[:, joint_index_start:], FLAGS.mean_relative)

        rootrel_absdiff = tf.abs((coords3d_true_rootrel - coords3d_pred_rootrel) / 1000)
        losses.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, inps.joint_validity_mask[:, joint_index_start:])

        if FLAGS.scale_agnostic_loss:
            _, scale_true = tfu.mean_stdev_masked(
                inps.coords3d_true[:, joint_index_start:, :], inps.joint_validity_mask[:, joint_index_start:], items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true[:, joint_index_start:, :]))
            _, scale_pred = tfu.mean_stdev_masked(
                preds.coords3d_pred_abs, inps.joint_validity_mask[:, joint_index_start:], items_axis=1, dimensions_axis=2,
                fixed_ref=tf.zeros_like(inps.coords3d_true[:, joint_index_start:, :]))
            preds.coords3d_pred_abs = tf.math.divide_no_nan(
                preds.coords3d_pred_abs, scale_pred) * scale_true

        if self.global_step > 5000:
            absdiff = tf.abs((inps.coords3d_true[:, joint_index_start:, :] - preds.coords3d_pred_abs) / 1000)
            losses.loss3d_abs = tfu.reduce_mean_masked(absdiff, inps.joint_validity_mask[:, joint_index_start:])
        else:
            losses.loss3d_abs = tf.constant(0, tf.float32)

        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
        
        losses.loss23d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true[:, joint_index_start:, :] - preds.coords2d_pred) * scale_2d),
            inps.joint_validity_mask[:, joint_index_start:])
        
        joint_index_start = 6 if FLAGS.output_upper_joints else 0
    
        preds.coords32d_pred_2d = models.util.align_2d_skeletons(
            preds.coords32d_pred_2d, inps.coords2d_true_2d[:, joint_index_start:, :], inps.joint_validity_mask_2d[:, joint_index_start:])
        losses.loss32d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d[:, joint_index_start:, :] - preds.coords32d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d[:, joint_index_start:])
        losses.loss22d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true_2d[:, joint_index_start:, :] - preds.coords22d_pred_2d) * scale_2d),
            inps.joint_validity_mask_2d[:, joint_index_start:])

        losses3d = [losses.loss3d, losses.loss23d, FLAGS.absloss_factor * losses.loss3d_abs]
        losses2d = [losses.loss22d, losses.loss32d]
        losses.loss = tf.add_n(losses3d) + FLAGS.loss2d_factor * tf.add_n(losses2d)
        
        return losses

    def compute_metrics(self, inps, preds):
#         return models.eval_metrics.compute_pose3d_metrics(inps, preds)
    
        return models.eval_metrics.compute_pose3d_metrics_j8(inps, preds) if FLAGS.output_upper_joints else models.eval_metrics.compute_pose3d_metrics(inps, preds)

    def forward_test(self, inps):

        preds = self._shared_process_inps (inps, training=False)

        if FLAGS.transform_coords:
            l2j = self.model.latent_points_to_joints
            preds.coords2d_pred = l2j(preds.coords2d_pred)
            preds.coords3d_rel_pred = l2j(preds.coords3d_rel_pred)
            preds.coords3d_pred_abs = l2j(preds.coords3d_pred_abs)

        return preds
