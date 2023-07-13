from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2
import tensorflow as tf
import time
from PyQt5.QtCore import Qt

# Other necessary imports and functions are supposed to be defined here.
# ...

lines = [] 
lines_2 = []
edge = [[9, 10], [9, 8], [6, 5], [12, 11], [12, 13], [4, 5], [4, 0], [11, 8], [8, 14], [8, 7], [0, 1], [0, 7], [3, 2], [15, 14], [15, 16], [1, 2]]
upper_skeleton = [(1, 2), (2, 3), (4, 5), (5, 6), (1,4)]  # head

def keyPressEvent(event):
    # Rotate the view
    if event.key() == Qt.Key_Left:
        window.orbit(-10, 0)  # Rotate left
    elif event.key() == Qt.Key_Right:
        window.orbit(10, 0)  # Rotate right
    elif event.key() == Qt.Key_Up:
        window.orbit(0, -10)  # Rotate up
    elif event.key() == Qt.Key_Down:
        window.orbit(0, 10)  # Rotate down

    # Pan the view
    elif event.key() == Qt.Key_W:
        window.pan(0, 1, 0)  # Pan up
    elif event.key() == Qt.Key_A:
        window.pan(-1, 0, 0)  # Pan left
    elif event.key() == Qt.Key_S:
        window.pan(0, -1, 0)  # Pan down
    elif event.key() == Qt.Key_D:
        window.pan(1, 0, 0)  # Pan right

    # Zoom the view
    elif event.key() == Qt.Key_Z:
        window.scale(0.9, 0.9, 0.9)  # Zoom in
    elif event.key() == Qt.Key_X:
        window.scale(1.1, 1.1, 1.1)  # Zoom out


def pred_pose3d(ml, frame):
    skeleton='h36m_17'
    pred = ml.detect_poses(frame, default_fov_degrees=55, skeleton=skeleton)
    return pred


def main():


#    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])
    camR = np.array([[[1., 0, 0], [0, 1, 0.], [0, 0., 1]]])

    use_pretrained_model = False
    use_keypoint_bbox = True
    input_size = 160


    app = QApplication([])
    app.keyPressEvent = keyPressEvent

    window = gl.GLViewWidget()
    window.resize(1024, 800)  # Set the desired window size
    window.show()
    window.setWindowTitle('Metraps demo')
    x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [10,0,0]]), color=pg.glColor((255, 0, 0, 255)), width=2, antialias=True)
    y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,10,0]]), color=pg.glColor((0, 255, 0, 255)), width=2, antialias=True)
    z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,10]]), color=pg.glColor((0, 0, 255, 255)), width=2, antialias=True)

    window.addItem(x_axis)
    window.addItem(y_axis)
    window.addItem(z_axis)

    # Add a grid on the X-Z plane
    g = gl.GLGridItem()
    g.setSize(x=40, y=40, z=40)
    g.setSpacing(x=1, y=1, z=1)
    window.addItem(g)

    # create scatter plot item
    scatter_plot_item = gl.GLScatterPlotItem(pxMode=False)
    window.addItem(scatter_plot_item)

    scatter_plot_item2 = gl.GLScatterPlotItem(pxMode=False)
    window.addItem(scatter_plot_item2)


    # Load the models
    if use_pretrained_model == True:
        print("Loading pretrained model")
        ## model : gold color
        model = tf.saved_model.load("metrabs_rn101_y4")  #metrabs_mob3s_y4

        ## model 2 : red color
        model2 = tf.saved_model.load("metrabs_rn18_y4")  #metrabs_mob3s_y4

    else:
        model = tf.saved_model.load("large_model/") 
        model2 = tf.saved_model.load("metrabs_rn101_y4")  #metrabs_mob3s_y4
        model_d = tf.saved_model.load("model_multi/")

    print("loaded pose estimation model")

    # Timer to update the scatter plot item's data every 100 ms

    def update():
        if use_pretrained_model:
            scatter_plot_item.setData(pos=pose3d[0], size=0.6, color=(1, 1, 1, 1), pxMode=False)
            scatter_plot_item2.setData(pos=pose3d_2[0], size=0.6, color=(1, 1, 1, 1), pxMode=False)
        else:
            scatter_plot_item.setData(pos=pose3d, size=0.6, color=(1, 1, 1, 1), pxMode=False)
            scatter_plot_item2.setData(pos=pose3d_2[0], size=0.6, color=(1, 1, 1, 1), pxMode=False)

        for line in lines:
            window.removeItem(line)

        for line in lines_2:
            window.removeItem(line)

        lines.clear()
        lines_2.clear()
        
        # Draw lines for skeleton connection

        if use_pretrained_model:
            for pair in edge:
                start, end = pair
                line_points = [pose3d[0][start], pose3d[0][end]]
                line = gl.GLLinePlotItem(pos=np.array(line_points), color = pg.glColor((255, 215, 0, 255)) , width=2, antialias=True)
                line_points_2 = [pose3d_2[0][start], pose3d_2[0][end]]
                line_2 = gl.GLLinePlotItem(pos=np.array(line_points_2), color = pg.glColor((255, 0, 0, 255)) , width=2, antialias=True)

                window.addItem(line)
                lines.append(line)            
                window.addItem(line_2)
                lines_2.append(line_2)        
        else:
            for pair in upper_skeleton:
                start, end = pair
                line_points = [pose3d[start], pose3d[end]]
                line = gl.GLLinePlotItem(pos=np.array(line_points), color = pg.glColor((255, 215, 0, 255)) , width=2, antialias=True)

                window.addItem(line)
                lines.append(line)            

            for pair in edge:
                start, end = pair
                line_points_2 = [pose3d_2[0][start], pose3d_2[0][end]]
                line_2 = gl.GLLinePlotItem(pos=np.array(line_points_2), color = pg.glColor((255, 0, 0, 255)) , width=2, antialias=True)

                window.addItem(line_2)
                lines_2.append(line_2)        


    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(100)  # update every 100 ms

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

    frame_counter = 0
    check_frame_number = 0
    start_time = time.time()


    for frame in frames_from_webcam():
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1: # for every second
            frame_rate = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()
        
                
        # if use_pretrained_model == True:
        #     pred = model.detect_poses(frame, skeleton='h36m_17')
        #     global pose3d            
        #     pose3d = pred['poses3d'].numpy()
        #     if pose3d.shape[0] == 0:
        #         continue            
        #     pose3d -= pose3d[:, 0, np.newaxis]
        #     pose3d = pose3d/50.0
        #     edge = model.per_skeleton_joint_edges['h36m_17'].numpy()

        
        global pose3d         
        global pose3d_2              


        if use_pretrained_model == True:
            
            if check_frame_number == 0:
                # skeleton='h36m_17'
                # pred = model.detect_poses(frame, default_fov_degrees=55, skeleton=skeleton)
                # pose3d = pred['poses3d'].numpy()

                pred = pred_pose3d(model, frame)
                pose3d = pred['poses3d'].numpy()
                if pose3d.shape[0] == 0:
                    continue            
                pose3d -= pose3d[:, 0, np.newaxis]
                pose3d = pose3d/50.0

                # pred_2 = model2.detect_poses(frame, default_fov_degrees=55, skeleton=skeleton)
                # pose3d_2 = pred_2['poses3d'].numpy()

                pred_2 = pred_pose3d(model2, frame)
                pose3d_2 = pred_2['poses3d'].numpy()

                if pose3d_2.shape[0] == 0:
                    continue            
                pose3d_2 -= pose3d_2[:, 0, np.newaxis]
                pose3d_2 = pose3d_2/50.0

            else:
#                print("we are using previous bbox")
                skeleton='h36m_17'
                keypoints = pred['poses2d'][0]
                xmin, ymin, xmax, ymax = 99999, 99999, -1, -1
                for kpt in keypoints:
                    xmin = min(xmin, kpt[0])
                    ymin = min(ymin, kpt[1])
                    xmax = max(xmax, kpt[0])
                    ymax = max(ymax, kpt[1])
                xmin = max(xmin - 20, 0)
                ymin = max(ymin - 80, 0)
                xmax = min(xmax + 20, frame.shape[1])
                ymax = min(ymax + 20, frame.shape[0])
                bbox = tf.convert_to_tensor(np.array([[xmin, ymin, xmax - xmin, ymax - ymin]], dtype=np.float32))

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 215), 2)

                pred = model.estimate_poses(frame, bbox, default_fov_degrees=55, skeleton=skeleton)
                pose3d = pred['poses3d'].numpy()
                if pose3d.shape[0] == 0:
                    continue            
                pose3d -= pose3d[:, 0, np.newaxis]
                pose3d = pose3d/50.0
                pred['boxes'] = bbox


                keypoints_2 = pred_2['poses2d'][0]
                xmin2, ymin2, xmax2, ymax2 = 99999, 99999, -1, -1
                for kpt in keypoints_2:
                    xmin2 = min(xmin2, kpt[0])
                    ymin2 = min(ymin2, kpt[1])
                    xmax2 = max(xmax2, kpt[0])
                    ymax2 = max(ymax2, kpt[1])
                xmin2 = max(xmin2 - 20, 0)
                ymin2 = max(ymin2 - 80, 0)
                xmax2 = min(xmax2 + 20, frame.shape[1])
                ymax2 = min(ymax2 + 20, frame.shape[0])
                bbox_2 = tf.convert_to_tensor(np.array([[xmin2, ymin2, xmax2 - xmin2, ymax2 - ymin2]], dtype=np.float32))

                cv2.rectangle(frame, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (0, 0, 255), 2)

                pred_2 = model2.estimate_poses(frame, bbox_2, default_fov_degrees=55, skeleton=skeleton)
                pose3d_2 = pred_2['poses3d'].numpy()
                if pose3d_2.shape[0] == 0:
                    continue            
                pose3d_2 -= pose3d_2[:, 0, np.newaxis]
                pose3d_2 = pose3d_2/50.0
                pred_2['boxes'] = bbox_2
        else:


            bboxes  = model_d.detector.predict_single_image(frame)
            if bboxes is not None: # draw bounding boxes if prediction is valid
                for bbox in bboxes:
                    x, y, w, h, score = bbox
                    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            crop_img, _ = adjust_bbox_and_get_crop(frame, x,y,w,h, upbb=False, mode=0)
            crop_img = cv2.GaussianBlur(crop_img, (25, 25), 0)             
#            crop_img = cv2.GaussianBlur(crop_img, (51, 51), 0)             

            pose3d, _ = get_pred(crop_img, input_size ,model, camR)
            pose3d = pose3d/50.0
#            pose_img = upper_plot_skeleton_cv2(pose, skeleton)

            pred_2 = pred_pose3d(model2, frame)
            pose3d_2 = pred_2['poses3d'].numpy()

            if pose3d_2.shape[0] == 0:
                continue            
            pose3d_2 -= pose3d_2[:, 0, np.newaxis]
            pose3d_2 = pose3d_2/50.0






        check_frame_number +=1

        cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    QApplication.instance().exec_()





def get_pred(img_crop, in_size, mdl, camera_rotate=None):
    crop_resize = cv2.resize(img_crop, (in_size, in_size), interpolation=cv2.INTER_CUBIC)
    inp = crop_resize.astype(np.float16) / 256.
    test = mdl(inp[np.newaxis, ...], False)
    test -= test[:, -1, np.newaxis]
    tt = test[0, :, :]  # (1,17,3)  cam.R (1,3,3) cam.t (1,3,1)
    return tt @ camera_rotate[0].T if camera_rotate is not None else tt, crop_resize


def frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


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



if __name__ == '__main__':
    main()

