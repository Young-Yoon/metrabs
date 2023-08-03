import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tf2onnx
import os

def convert_to_pb(concrete_func, output_path):
    # Convert the concrete function to a frozen function
    frozen_func = convert_variables_to_constants_v2(concrete_func)

    # Save the frozen function to .pb format
    frozen_graph_def = frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph=frozen_graph_def, logdir=output_path, name="metro_mbv3_l(f).pb", as_text=False)

def convert_to_tflite(concrete_func, output_path):
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    with open(os.path.join(output_path, 'metro_mbv3_l(f).tflite'), 'wb') as f:
        f.write(tflite_model)

def convert_to_onnx(concrete_func, output_path):
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_function(
        func=concrete_func,
        opset=13,  # Change the opset version if needed
        output_path=os.path.join(output_path, 'metro_mbv3_l(f).onnx')
    )

def convert_model(model_path, output_path, output_format):
    # Load the model
    loaded = tf.saved_model.load(model_path)

    # Get the concrete function from the loaded model
    concrete_func = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    if output_format in ('pb', 'all'):
        convert_to_pb(concrete_func, output_path)

    if output_format in ('tflite', 'all'):
        convert_to_tflite(concrete_func, output_path)

    if output_format in ('onnx', 'all'):
        convert_to_onnx(concrete_func, output_path)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Load a TensorFlow model and save it in different formats.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model to be loaded.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output files.')
    parser.add_argument('--output_format', type=str, choices=['pb', 'tflite', 'onnx', 'all'], default='all', help='Output format(s) to convert the model into.')
    args = parser.parse_args()

    convert_model(args.model_path, args.output_path, args.output_format)


if __name__ == '__main__':
    main()
