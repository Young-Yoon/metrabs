import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('-f', '--folder_names', type=str, nargs='+', required=True, help='List of folder names')
    parser.add_argument('-v', '--view', type=str, choices=['3d', 'front', 'side', 'all'], default='all', help='Which views to plot')
    return parser.parse_args()

def load_model(model_path):
    return tf.saved_model.load(model_path)

def get_images(folder_name):
    return sorted(glob.glob(os.path.join(folder_name, '*.jpg')))

def predict_and_plot(model, image_files, view):
    for k in range(len(image_files)):    
        file_name = image_files[k]
        image = tf.image.decode_jpeg((tf.io.read_file(image_files[k])))
        a, b = file_name.split(".jpg")
        image2 = tf.image.resize(image, [256,256])
        crops = tf.stack([image2, image2], axis=0)[:, :256, :256]
        crops = tf.cast(crops, tf.float16)/255
        poses3d = model.predict_multi(crops)
        plot_figures(image, np.array(poses3d[0]), a, view)

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('-f', '--folder_names', type=str, nargs='+', required=True, help='List of folder names')
    parser.add_argument('-v', '--view', type=str, choices=['3d', 'front', 'side', 'all'], default='all', help='Which views to plot')
    return parser.parse_args()

def load_model(model_path):
    return tf.saved_model.load(model_path)

def get_images(folder_name):
    return sorted(glob.glob(os.path.join(folder_name, '*.jpg')))

def predict_and_plot(model, image_files, view):
    for k in range(len(image_files)):    
        file_name = image_files[k]
        image = tf.image.decode_jpeg((tf.io.read_file(image_files[k])))
        a, b = file_name.split(".jpg")
        image2 = tf.image.resize(image, [256,256])
        crops = tf.stack([image2, image2], axis=0)[:, :256, :256]
        crops = tf.cast(crops, tf.float16)/255
        poses3d = model.predict_multi(crops)
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
    plt.savefig(file_name+ ".png")
    plt.clf()
    plt.close('all')


def main():
    args = get_args()
    model_path = args.model_path
    folder_names = args.folder_names
    view = args.view
    model = load_model(model_path)
    if model is None:
        return
    for folder_name in folder_names:
        image_files = get_images(folder_name)
        predict_and_plot(model, image_files, view)

if __name__ == "__main__":
    main()


def main():
    args = get_args()
    model_path = args.model_path
    folder_names = args.folder_names
    view = args.view
    model = load_model(model_path)
    if model is None:
        return
    for folder_name in folder_names:
        image_files = get_images(folder_name)
        predict_and_plot(model, image_files, view)

if __name__ == "__main__":
    main()
