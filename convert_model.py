import tensorflow as tf
model_folder = 'models/metrabs_eff2s_y4/'
out_fold = 'models/eff2s_y4_short_sig'
model = tf.saved_model.load(model_folder)


@tf.function()
def my_predict(my_prediction_inputs, **kwargs):
    prediction = model.detect_poses(my_prediction_inputs)
    return {"prediction": prediction['poses3d']}


# add signature
my_signatures = my_predict.get_concrete_function(
   my_prediction_inputs=tf.TensorSpec([None, None, 3], dtype=tf.dtypes.uint8, name="image"))

tf.saved_model.save(model, out_fold, signatures=my_signatures)
