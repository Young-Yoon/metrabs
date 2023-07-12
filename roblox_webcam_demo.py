# install tensorflow  ( window-native)
#https://www.tensorflow.org/install/pip#windows-native

import cv2
import time
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
from PIL import Image

def upper_plot_skeleton_cv2(p, color='r'):

    rotation_angle = np.pi / 2  # 90 degrees in radians

    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    pose_3d_rotated = np.dot(p, rotation_matrix)

    p =  np.array(p)


    p_2d = p[:, :2]
    p_2d[:,1] = -p[:, 2]

    p_2d_r = pose_3d_rotated[:, :2]
    p_2d_r[:,1] = -pose_3d_rotated[:, 2]

    p_2d =  np.array(p_2d)
    # Scale and shift points to fit into the image
    p_2d -= p_2d.min(axis=0)
    p_2d /= p_2d.ptp(axis=0)
    p_2d *= 100

    # Scale and shift points to fit into the image
    p_2d_r -= p_2d_r.min(axis=0)
    p_2d_r /= p_2d_r.ptp(axis=0)
    p_2d_r *= 100



# Add an offset of 20 pixels to x and y coordinates
    p_2d[:, 0] += 20
    p_2d[:, 1] += 20    

    p_2d_r[:, 0] += 150
    p_2d_r[:, 1] += 20    

    p_2d = p_2d.astype(int)
    p_2d_r = p_2d_r.astype(int)

    # Create an image to draw on
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Define connections between joints
    skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1,4)]  # head
    center_shoulder_index = len(p_2d)  # Index of the new point
    p_2d = np.vstack([p_2d, (p_2d[1] + p_2d[4]) / 2])  # Add the center shoulder point
    p_2d_r = np.vstack([p_2d_r, (p_2d_r[1] + p_2d_r[4]) / 2])  # Add the center shoulder point

    skeleton += [(center_shoulder_index, 0), (center_shoulder_index, 7)]

    # Draw the skeleton
    for i, j in skeleton:
        cv2.line(img, tuple(map(int, p_2d[i])), tuple(map(int, p_2d[j])), (0, 255, 0), 2)
        cv2.line(img, tuple(map(int, p_2d_r[i])), tuple(map(int, p_2d_r[j])), (0, 255, 0), 2)

    return img


def upper_plot_skeleton_plt(p, color='r'):
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    deg = 0
    views = [(deg, deg - 90), (deg, deg), (90 - deg, deg - 90)]    
    axis.view_init(*views[0])     

    axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color)

    skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1,4)]  # head
    axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color, alpha=1.0 )

    for i, j in skeleton:
        axis.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], [p[i, 2], p[j, 2]], color)            

    center_shoulder = (p[1,:] + p[4,:])/2.0     
    axis.plot([center_shoulder[0], p[0, 0]], [center_shoulder[1], p[0, 1]], [center_shoulder[2], p[0, 2]], color)            
    axis.plot([center_shoulder[0], p[7, 0]], [center_shoulder[1], p[7, 1]], [center_shoulder[2], p[7, 2]], color)           
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    axis.set_xlim3d(-700, 700)
    axis.set_zlim3d(-700, 700)
    axis.set_ylim3d(-700, 700)

    return fig


def get_pred(img_crop, in_size, mdl, camera_rotate=None):
    crop_resize = cv2.resize(img_crop, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
    inp = crop_resize.astype(np.float16) / 256.
    test = mdl(inp[np.newaxis, ...], False)
    test -= test[:, -1, np.newaxis]
    tt = test[0, :, :]  # (1,17,3)  cam.R (1,3,3) cam.t (1,3,1)
    return tt @ camera_rotate[0].T if camera_rotate is not None else tt, crop_resize



def adjust_bbox_and_get_crop(img0, x0, y0, w0, h0, upbb=False, mode=0): 


    if upbb:
        h0 *= 0.5
    if w0 < h0:
        w1, h1 = h0, h0
        x1, y1 = x0 - (h0 - w0) / 2., y0
    else:
        w1, h1 = w0, w0  
        x1, y1 = x0, y0 - (w0 - h0) / 2.    
    
    x1_max = x1 + w1
    y1_max = y1 + h1
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x1_max = min(x1_max, img0.shape[1])
    y1_max = min(y1_max, img0.shape[0])

    # x1_max = min(x1_max, img0.width)
    # y1_max = min(y1_max, img0.height)  

    w1 = x1_max - x1
    h1 = y1_max - y1
    
    if mode in [0, 1]:    
        crop_img = np.array(img0)[int(y1):int(y1_max), int(x1):int(x1_max)]
        return crop_img, (x1, y1, w1, h1)
    elif mode == 2:
        crop_img = np.array(img0)[int(y1):int(y1_max), int(x1):int(x1_max)]
        crop_h, crop_w, _ = crop_img.shape
        crop_l = max(crop_h, crop_w)
        zp_crop_img = np.zeros([crop_l, crop_l, 3], dtype=crop_img.dtype)
        h_offset = int((crop_l - crop_h) / 2)
        w_offset = int((crop_l - crop_w) / 2)
        zp_crop_img[h_offset: h_offset + crop_h, w_offset: w_offset + crop_w] = crop_img
        return zp_crop_img, (x1, y1, w1, h1)
    elif mode == 3:
        if w1 < h1:
            w3, h3 = w1, w1
            x3, y3 = x1, y1 + (h1 - w1) * 0.1
        else:
            w3, h3 = h1, h1  
            x3, y3 = x1 + (w1 - h1) * 0.5, y1
        x3_max = x3 + w3
        y3_max = y3 + h3
        crop_img = np.array(img0)[int(y3):int(y3_max), int(x3):int(x3_max)]
        return crop_img, (x3, y3, w3, h3)
    else:
        print("Error. Unsupported bbox squarization mode. Only 0, 1, 2, 3 are supported!")
        exit()


def image_test():

    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])

    model = tf.saved_model.load("model/")
    print("loaded pose estimation model")
    model_d = tf.saved_model.load("model_multi/")
    print("loaded bbox detection model")

    frames = sorted(glob.glob(os.path.join('inaki_boxing1/*.jpg')))
    frames = [f.split('/')[-1] for f in frames]
    total_frames = len(frames)
    if total_frames == 0:
        print(f'No frames in the folder')

    print(total_frames)
    deg = 0
    views = [(deg, deg - 90), (deg, deg), (90 - deg, deg - 90)]
    fsz = 2

    for i_fr, frame in enumerate(tqdm(frames)):
        save_path = 'output/' + frame
        #print(save_path, input_dir, frame)
        input_file = os.path.join(frame)
        print(input_file)
        img = Image.open(input_file)
        bbox = model_d.detector.predict_single_image(np.array(img))
        x, y, wd, ht, conf = bbox[0]
        crop_sq, crop_bbox = adjust_bbox_and_get_crop(np.array(img), x, y, wd, ht, mode=3)
        x_sq, y_sq, wd_sq, ht_sq = crop_bbox
        tt_sq, res_sq = get_pred(crop_sq, 160, model, camR)



        fig = plt.figure()
        axis = fig.add_subplot(122, projection='3d')

        p = tt_sq
        color='r'

        axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color)

        skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1,4)]  # head
        axis.view_init(*views[0])     
        axis.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=color, alpha=1.0 )

        for i, j in skeleton:
            axis.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], [p[i, 2], p[j, 2]], color)            

        center_shoulder = (p[1,:] + p[4,:])/2.0     
        axis.plot([center_shoulder[0], p[0, 0]], [center_shoulder[1], p[0, 1]], [center_shoulder[2], p[0, 2]], color)            
        axis.plot([center_shoulder[0], p[7, 0]], [center_shoulder[1], p[7, 1]], [center_shoulder[2], p[7, 2]], color)           
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        axis.set_zlabel('Z')

        axis.set_xlim3d(-700, 700)
        axis.set_zlim3d(-700, 700)
        axis.set_ylim3d(-700, 700)

        ax3 = fig.add_subplot(1, 2, 1)
        ax3.imshow(res_sq)
        plt.savefig(save_path)
        plt.close()

        
def main():

    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])

    input_size = 112
    model = tf.saved_model.load("low_model/")
    print("loaded pose estimation model")
    model_d = tf.saved_model.load("model_multi/")
    print("loaded bbox detection model")

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)  # New window for cropped images
    cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)  # New window for pose


    frame_counter = 0
    start_time = time.time()


    for frame in frames_from_webcam():
        frame_counter += 1
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 1: # for every second
            frame_rate = frame_counter / elapsed_time
#            print("Frame rate:", frame_rate)
            
            # Reset counter and time
            frame_counter = 0
            start_time = time.time()
        
        bboxes  = model_d.detector.predict_single_image(frame)

        if bboxes is not None: # draw bounding boxes if prediction is valid
            for bbox in bboxes:
                x, y, w, h, score = bbox
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        crop_img, _ = adjust_bbox_and_get_crop(frame, x,y,w,h, upbb=True, mode=3)
                
        pose, _ = get_pred(crop_img, input_size ,model, camR)

        pose_img = upper_plot_skeleton_cv2(pose)

        cv2.imshow('Pose', pose_img)  # Display the cropped image
        cv2.imshow('Cropped', crop_img)  # Display the cropped image
        cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

if __name__ == '__main__':
    main()