import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import onnx
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('-f', '--folder_names', type=str, nargs='+', required=True, help='List of folder names')
    parser.add_argument('-v', '--view', type=str, choices=['3d', 'front', 'side', 'all'], default='all', help='Which views to plot')
    return parser.parse_args()


def load_model(model_path, model_type):
    if model_type == 'tensorflow':
        return tf.saved_model.load(model_path)
    elif model_type == 'onnx':
        return ort.InferenceSession(model_path)


def get_images(folder_name):
    return sorted(glob.glob(os.path.join(folder_name, '*.jpg')))


def run_inference(model, image):
    if isinstance(model, tf.Graph):
        with tf.compat.v1.Session(graph=model) as sess:
            # Assuming that 'input:0' is the name of the input tensor and 'output:0' is the name of the output tensor.
            output_tensor = model.get_tensor_by_name('Identity:0')
            input_tensor = model.get_tensor_by_name('input_2:0')

            image2 = tf.image.resize(image, [256, 256])
            crops = tf.stack([image2, image2], axis=0)[:, :256, :256]
            crops = tf.cast(crops, tf.float16) / 255

            # Run inference using the TensorFlow session
            predictions = sess.run(output_tensor, feed_dict={input_tensor: crops})
            return predictions

    elif isinstance(model, ort.InferenceSession):
        # Get the names of the input and output nodes
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name

        # Preprocess the image
        image = image.resize((256, 256))
        image = np.array(image).astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # Run inference using the ONNX runtime session
        predictions = model.run([output_name], {input_name: image})[0]
        return predictions


def predict_and_plot(model, image_files, view):
    for k in range(len(image_files)):
        file_name = image_files[k]
        image = tf.image.decode_jpeg(tf.io.read_file(image_files[k]))
        a, _ = os.path.splitext(os.path.basename(file_name))

        # Run inference
        predictions = run_inference(model, image)

        # Perform further processing and visualization using the `predictions` variable
        # Adjust this part according to your specific requirements and model outputs
        poses3d = predictions  # Example assignment, modify as needed

        # Visualize the predictions
        plot_figures(image, np.array(poses3d[0]), a, view)


def plot_figures(image, tt, file_name, view='all'):
    skeleton = [(9, 8), (8, 7), (7, 10), (10, 11), (11, 12),
                (7, 13), (13, 14), (14, 15), (7, 6), (6, 16),
                (16, 3), (3, 4), (4, 5), (16, 0), (0, 1), (1, 2)]
    fig = plt.figure(figsize=(10, 4))
    ax2 = fig.add_subplot(141)
    ax2.imshow(image)
    if view in ('3d', 'all'):
        ax = fig.add_subplot(142, projection='3d')
        ax.scatter(tt[:, 0], tt[:, 1], tt[:, 2])
        for i, j in skeleton:
            plt.plot([tt[i, 0], tt[j, 0]], [tt[i, 1], tt[j, 1]], [tt[i, 2], tt[j, 2]], 'r')
        ax.set_title('3D View')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim3d(-500, 500)
        ax.set_zlim3d(-1000, 1000)
        ax.set_ylim3d(-500, 500)
    if view in ('front', 'all'):
        ax3 = fig.add_subplot(143, projection='3d')
        ax3.view_init(5, -85)
        ax3.scatter(tt[:, 0], tt[:, 1], tt[:, 2])
        for i, j in skeleton:
            plt.plot([tt[i, 0], tt[j, 0]], [tt[i, 1], tt[j, 1]], [tt[i, 2], tt[j, 2]], 'r')
        ax3.set_title('Front View')
        ax3.set_xlabel('X Label')
        ax3.set_xlim3d(-500, 500)
        ax3.set_zlim3d(-1000, 1000)
        ax3.set_ylim3d(-500, 500)
    if view in ('side', 'all'):
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.view_init(0, 0)
        ax4.scatter(tt[:, 0], tt[:, 1], tt[:, 2])
        for i, j in skeleton:
            plt.plot([tt[i, 0], tt[j, 0]], [tt[i, 1], tt[j, 1]], [tt[i, 2], tt[j, 2]], 'r')
        ax4.set_title('Side View')
        ax4.set_ylabel('Y Label')
        ax4.set_xlim3d(-500, 500)
        ax4.set_zlim3d(-1000, 1000)
        ax4.set_ylim3d(-500, 500)
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.clf()
    plt.close('all')


def process_images(model_path, folder_names, view):
    model_type = model_path.split('.')[-1]
    model = load_model(model_path, model_type)
    if model is None:
        return
    for folder_name in folder_names:
        image_files = get_images(folder_name)
        predict_and_plot(model, image_files, view)


def main():
    args = get_args()
    model_path = args.model_path
    folder_names = args.folder_names
    view = args.view
    process_images(model_path, folder_names, view)


if __name__ == "__main__":
    main()
