#!/usr/bin/env python3
import tensorflow as tf
model_folder = "/home/jovyan/runs/metrabs-exp/models/eff2s_y4/model_multi"
out_folder = model_folder+"/detector"
model = tf.saved_model.load(model_folder)
print(model.detector.model.__dir__())
det = model.detector
tf.saved_model.save(model.detector, out_folder)
