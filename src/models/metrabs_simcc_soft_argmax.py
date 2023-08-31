import einops
import keras
import keras.layers
import keras.metrics
import numpy as np
import tensorflow as tf
from attrdict import AttrDict
from tensorflow.keras.layers import Dense, Flatten, Reshape

import models.eval_metrics
import models.model_trainer
import models.metrabs
import models.util
import tfu
import tfu3d
from options import FLAGS

    
def softargmax1d(input, depth,  indice, beta=1 ):
    input = tf.nn.softmax(beta * input, axis=-1)
    indices = tf.linspace(0.0, 1.0, indice)
    result = tf.reduce_sum(depth * input * indices, axis=-1)
    return result
    
class MetrabsSimCCSoftArgMaxHeads(models.metrabs.MetrabsHeads):
    def __init__(self, n_points):
        super().__init__(n_points)
        
        self.fc_t_layer = Dense(FLAGS.depth_length*3) ## simplified version by concatenating all 3 dimensions in one layer 
        self.fc_t2_layer = Dense(FLAGS.proc_side*2)## simplified version by concatenating 2 dimensions in one layer 
        

    def call(self, inp, training=None):
        x = self.conv_final(inp)
        logits2d, logits3d = tf.split(x, self.n_outs, axis=tfu.channel_axis())

        logits_shape = tf.shape(logits2d)
        b = logits_shape[0]  # Get the dynamic batch size          

        ## This currently only supports the MobileNetV3 backbone 
        reshape_factor_2d = FLAGS.proc_side // 32
        logits2d = tf.reshape(logits2d, (b, self.n_points, reshape_factor_2d * reshape_factor_2d))
        logits3d = tf.reshape(logits3d, (b, self.n_points, FLAGS.depth * 8* 8))
        
        outputs3d = []
        outputs2d = []
        
        for i in range(self.n_points):
            new_logits3d = logits3d[:,i,:]
            new_logits2d = logits2d[:,i,:]

            output3d_2 = self.fc_t_layer(new_logits3d)  # (None, 384)               
            output3d_2 = tf.reshape(output3d_2, (b, FLAGS.depth_length, 3))  # shape: (None, 128, 3)

            output2d_2 = self.fc_t2_layer(new_logits2d)  # (None, 512)               
            output2d_2 = tf.reshape(output2d_2, (b, FLAGS.proc_side, 2))  # shape: (None, 256, 2)
    
            outputs3d.append(output3d_2)
            outputs2d.append(output2d_2)

        outputs3d = tf.stack(outputs3d, axis=-2) # shape: (None, 128, 8, 3)
        outputs2d = tf.stack(outputs2d, axis=-2) # shape: (None, 256, 8, 2)

        outputs3d = tf.reshape(outputs3d, (b, 8, 3, -1))  # shape: (None, 8, 3, 128)
        outputs2d = tf.reshape(outputs2d, (b, 8, 2, -1))  # shape: (None, 8, 2, 256)        
        

        return outputs2d , outputs3d
                
        


class MetrabsSimCCSoftArgMaxTrainer(models.metrabs.MetrabsTrainer):

    def forward_train(self, inps, training):        
        preds = AttrDict()

        image_both = tf.concat([inps.image, inps.image_2d], axis=0)
        features = self.model.backbone(image_both, training=training)
        coords2d_pred_both, coords3d_rel_pred_both = self.model.heatmap_heads(
            features, training=training)

        coords3d_rel_pred_both = softargmax1d(coords3d_rel_pred_both, FLAGS.box_size_mm, indice=FLAGS.depth_length)
        coords2d_pred_both = softargmax1d(coords2d_pred_both, FLAGS.proc_side, indice=256)

        
        batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]
        preds.coords2d_pred, preds.coords2d_pred_2d = tf.split(
            coords2d_pred_both, batch_sizes, axis=0)
        preds.coords3d_rel_pred, preds.coords3d_rel_pred_2d = tf.split(
            coords3d_rel_pred_both, batch_sizes, axis=0)


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
            
        input_coord3d = inps.coords3d_true[:, joint_index_start:, :]
        center = input_coord3d[:, -1:]
        input_coord3d = input_coord3d - center + FLAGS.box_size_mm/2
            
            
        rootrel_absdiff = tf.abs((input_coord3d - preds.coords3d_rel_pred) / 1000)
        losses.loss3d = tfu.reduce_mean_masked(rootrel_absdiff, inps.joint_validity_mask[:, joint_index_start:])

        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_mm / 1000
        
        losses.loss23d = tfu.reduce_mean_masked(
            tf.abs((inps.coords2d_true[:, joint_index_start:, :] - preds.coords2d_pred) * scale_2d),
            inps.joint_validity_mask[:, joint_index_start:])
        
        
        losses3d = [losses.loss3d, losses.loss23d]
        losses.loss = tf.add_n(losses3d) 
        
        return losses

    def compute_metrics(self, inps, preds):   
        return models.eval_metrics.compute_pose3d_metrics_simcc_soft_argmax(inps, preds)

    def forward_test(self, inps):
        preds = AttrDict()
        features = self.model.backbone(inps.image, training=False)
        preds.coords2d_pred, preds.coords3d_rel_pred = self.model.heatmap_heads(
            features, training=False)

        preds.coords3d_rel_pred = softargmax1d(preds.coords3d_rel_pred, FLAGS.box_size_mm, indice=FLAGS.depth_length)
        preds.coords2d_pred = softargmax1d(preds.coords2d_pred, FLAGS.proc_side, indice=256)


        
        return preds