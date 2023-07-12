from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2
import tensorflow as tf
import time

# Other necessary imports and functions are supposed to be defined here.
# ...

lines = [] 
edge = [[9, 10], [9, 8], [6, 5], [12, 11], [12, 13], [4, 5], [4, 0], [11, 8], [8, 14], [8, 7], [0, 1], [0, 7], [3, 2], [15, 14], [15, 16], [1, 2]]


def main():
    app = QApplication([])
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

    camR = np.array([[[1., 0, 0], [0, 0, 1.], [0, -1., 0]]])
    use_pretrained_model = True
    input_size = 160

    # Load the models
    if use_pretrained_model == True:
        print("Loading pretrained model")
        model = tf.saved_model.load("metrabs_rn101_y4")
    else:
        model = tf.saved_model.load("large_model/")

    
    print("loaded pose estimation model")

    # model_d = tf.saved_model.load("model_multi/")
    # print("loaded bbox detection model")

    # Timer to update the scatter plot item's data every 100 ms
    def update():
        if use_pretrained_model:
            scatter_plot_item.setData(pos=pose3d[0], size=1.0, color=(1, 1, 1, 1), pxMode=False)

        for line in lines:
            window.removeItem(line)
        lines.clear()
        
        # Draw lines for skeleton connection
        for pair in edge:
            start, end = pair
            line_points = [pose3d[0][start], pose3d[0][end]]
            line = gl.GLLinePlotItem(pos=np.array(line_points), color = pg.glColor((255, 255, 0, 255)) , width=2, antialias=True)
            window.addItem(line)
            lines.append(line)            

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(100)  # update every 100 ms


    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)  # New window for cropped images
    # cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)  # New window for pose


    frame_counter = 0
    start_time = time.time()


    for frame in frames_from_webcam():
        frame_counter += 1
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 1: # for every second
            frame_rate = frame_counter / elapsed_time
            frame_counter = 0
            start_time = time.time()
        
                
        if use_pretrained_model == True:
            pred = model.detect_poses(frame, skeleton='h36m_17')
            global pose3d            
            pose3d = pred['poses3d'].numpy()
            if pose3d.shape[0] == 0:
                continue            
            pose3d -= pose3d[:, 0, np.newaxis]
            pose3d = pose3d/50.0
            edge = model.per_skeleton_joint_edges['h36m_17'].numpy()

        cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    QApplication.instance().exec_()


def frames_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


if __name__ == '__main__':
    main()



