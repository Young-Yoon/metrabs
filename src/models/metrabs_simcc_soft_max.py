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


class MetrabsSimCCSoftMaxHeads(models.metrabs.MetrabsHeads):
    def __init__(self, n_points):
        super().__init__(n_points)

        self.fc1_layer = Dense(FLAGS.box_size_mm)
        self.fc2_layer = Dense(FLAGS.box_size_mm)
        self.fc3_layer = Dense(FLAGS.box_size_mm)

        self.fc4_layer = Dense(FLAGS.proc_side)
        self.fc5_layer = Dense(FLAGS.proc_side)

    def call(self, inp, training=None):
        x = self.conv_final(inp)
        logits2d, logits3d = tf.split(x, self.n_outs, axis=tfu.channel_axis())
        
        logits_shape = tf.shape(logits2d)
        b = logits_shape[0]  # Get the dynamic batch size          

        ## This currently only supports the MobileNetV3 backbone 
        reshape_factor_2d = FLAGS.proc_side // 32
        logits2d = tf.reshape(logits2d, (b, self.n_points,  reshape_factor_2d * reshape_factor_2d)) 
        logits3d = tf.reshape(logits3d, (b, self.n_points, FLAGS.depth * 8* 8))

        outputs3d_x = []
        outputs3d_y = []
        outputs3d_z = []

        outputs2d_x = []
        outputs2d_y = []

        for i in range(self.n_points):       
            new_logits3d = logits3d[:,i,:]
            new_logits2d = logits2d[:,i,:]

            output3d_x = self.fc1_layer(new_logits3d)  # desired output shape: (None, 2200)
            output3d_y = self.fc2_layer(new_logits3d)  # desired output shape: (None, 2200)
            output3d_z = self.fc3_layer(new_logits3d)  # desired output shape: (None, 2200)

            output2d_x = self.fc4_layer(new_logits2d) # (None, 256 )
            output2d_y = self.fc5_layer(new_logits2d) # (None, 256 )


            outputs3d_x.append(output3d_x)
            outputs3d_y.append(output3d_y)
            outputs3d_z.append(output3d_z)

            outputs2d_x.append(output2d_x)
            outputs2d_y.append(output2d_y)

        outputs3d_x = tf.stack(outputs3d_x, axis=-1)  # (None, 2200, 8)
        outputs3d_y = tf.stack(outputs3d_y, axis=-1)  # (None, 2200, 8)
        outputs3d_z = tf.stack(outputs3d_z, axis=-1)  # (None, 2200, 8)

        outputs2d_x = tf.stack(outputs2d_x, axis=-1)  # (None, 256, 8)
        outputs2d_y = tf.stack(outputs2d_y, axis=-1)  # (None, 256, 8)

        outputs3d = tf.stack([outputs3d_x, outputs3d_y, outputs3d_z], axis=-1)  # shape: (None, 2200, 8, 3)
        outputs2d = tf.stack([outputs2d_x, outputs2d_y], axis=-1)  # shape: (None, 256, 8, 2)

        return outputs2d , outputs3d


class MetrabsSimCCSoftMaxTrainer(models.metrabs.MetrabsTrainer):
    def forward_train(self, inps, training):        
        preds = AttrDict()

        image_both = tf.concat([inps.image, inps.image_2d], axis=0)
        features = self.model.backbone(image_both, training=training)
        outputs2d, outputs3d = self.model.heatmap_heads(
            features, training=training)
        
        batch_sizes = [t.shape.as_list()[0] for t in [inps.image, inps.image_2d]]

        preds.coords2d_pred, preds.coords2d_pred_2d = tf.split(
            outputs2d, batch_sizes, axis=0)
        preds.coords3d_rel_pred, preds.coords3d_rel_pred_2d = tf.split(
            outputs3d, batch_sizes, axis=0)        
        
        
        prob_output3d = tf.nn.softmax(preds.coords3d_rel_pred, axis=1)   # shape: (None, 2200, 8, 3)
        prob_output2d = tf.nn.softmax(preds.coords2d_pred, axis=1)       # shape: (None, 256, 8, 3)
 
        preds.coords3d_rel_pred = tf.transpose(prob_output3d, perm=[0, 2, 3, 1])   # shape: (None, 8, 3, 2200)
        preds.coords2d_pred = tf.transpose(prob_output2d, perm=[0, 2, 3, 1])       # shape: (None, 8, 2, 256)


        return preds

    def compute_losses(self, inps, preds):
        losses = AttrDict()
       
        joint_index_start = 9 if FLAGS.output_upper_joints else 0 
        
        kl_div_loss = tf.keras.losses.KLDivergence()
        
        input_coord3d = inps.coords3d_true[:, joint_index_start:, :]
        center = input_coord3d[:, -1:]
        input_coord3d = input_coord3d - center + FLAGS.box_size_mm/2
        
#         inps_coords3d_true_int = tf.cast(inps.coords3d_true[:, joint_index_start:, :], tf.int32)        
        inps_coords3d_true_int = tf.cast(input_coord3d, tf.int32)        
        
        coords3d_true_one_hot = tf.one_hot(inps_coords3d_true_int, FLAGS.box_size_mm )
        coords3d_true_one_hot_float = tf.cast(coords3d_true_one_hot, tf.float32)        

        
        inps_coords2d_true_int = tf.cast(inps.coords2d_true[:, joint_index_start:, :], tf.int32)        
        coords2d_true_one_hot = tf.one_hot(inps_coords2d_true_int, FLAGS.proc_side )
        coords2d_true_one_hot_float = tf.cast(coords2d_true_one_hot, tf.float32)        

                
        losses.loss_kl_3d = kl_div_loss(coords3d_true_one_hot_float, preds.coords3d_rel_pred)
        losses.loss_kl_2d = kl_div_loss(coords2d_true_one_hot_float, preds.coords2d_pred)

        losses3d = [ losses.loss_kl_3d, losses.loss_kl_2d]
        losses.loss = tf.add_n(losses3d) 
        
        return losses

    def compute_metrics(self, inps, preds):   
        return models.eval_metrics.compute_pose3d_metrics_simcc_soft_max(inps, preds)

    def forward_test(self, inps):
        preds = AttrDict()
        features = self.model.backbone(inps.image, training=False)
        preds.coords2d_pred, preds.coords3d_rel_pred = self.model.heatmap_heads(
            features, training=False)

        preds.coords3d_rel_pred = tf.transpose(preds.coords3d_rel_pred, perm=[0, 2, 3, 1])   # shape: (None, 8, 3, 2200)
        preds.coords2d_pred = tf.transpose(preds.coords2d_pred, perm=[0, 2, 3, 1])       # shape: (None, 8, 2, 256)

        
        return preds