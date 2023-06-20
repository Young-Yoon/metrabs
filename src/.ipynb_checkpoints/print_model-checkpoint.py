#!/usr/bin/env python3
import sys
import tensorflow as tf
model_folder = "/home/jovyan/runs/metrabs-exp/" + sys.argv[1] #models/eff2s_y4/model_multi"
out_folder = model_folder+"/detector"
model = tf.saved_model.load(model_folder)
print(model.__dir__())

#tf.saved_model.save(model.detector, out_folder)
