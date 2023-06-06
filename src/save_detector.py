#!/usr/bin/env python3
import tensorflow as tf
model_folder = "/home/jovyan/runs/metrabs-exp/models/eff2s_y4/model_multi"
out_folder = model_folder+"/detector"
model = tf.saved_model.load(model_folder)
print(model.detector.model.__dir__())
det = model.detector
tf.saved_model.save(model.detector, out_folder)

'''
# create blank effnet model
from backbones.efficientnet.effnetv2_model import *
import backbones.efficientnet.effnetv2_utils as effnet_util
import tfu
effnet_util.set_batchnorm(effnet_util.BatchNormalization)
tfu.set_data_format('NHWC')
tfu.set_dtype(tf.float16)
mod = get_model('efficientnetv2-s', include_top=False, pretrained=False, with_endpoints=False)

# copy over trained weights:
new_vars = mod.variables
var_dict = {v.name: [v, i] for i, v in enumerate(vars)}
var_dict_new = {v.name: [v, i] for i, v in enumerate(new_vars)}
inds = [var_dict[k][1] for k in var_dict_new.keys() if k in var_dict]
print(len(var_dict_new))
print(len(inds))

missing_keys = set(var_dict.keys()) - set(var_dict_new.keys())
rev_missing_keys = set(var_dict_new.keys()) - set(var_dict.keys())

print(missing_keys)
print(rev_missing_keys)
for m in missing_keys:
    d = var_dict[m][0]
    print(d.name, d.shape)
pick_vars = [vars[i] for i in inds]
print(len(pick_vars))

mod.set_weights(pick_vars)
@tf.function()
def my_predict(my_prediction_inputs, **kwargs):
    #print(mod.__dir__())
    prediction = det(my_prediction_inputs, training=False)
    return {"prediction": prediction}

my_signatures = my_predict.get_concrete_function(
   my_prediction_inputs=tf.TensorSpec([None, 256, 256, 3], dtype=tf.dtypes.float32, name="image")
)
'''
