import os.path
import numpy as np
import cameralib
import data.datasets3d as ps3d
import paths
import util
from data.preproc_for_efficiency import make_efficient_example


def vis(imagepath, projected_2d, bbox, keypoint_2d=None):
    import matplotlib.pyplot as plt
    from skimage.io import imread, imshow
    image = imread(imagepath)
    h, w, _ = image.shape
    plt.imshow(image)
    
    ### Show the reprojected keypoints
    for j in range(projected_2d.shape[0]):
        plt.plot(projected_2d[j, 0], projected_2d[j, 1], "o", markersize=7, color="orange")
        
    ### Show the keypoints
    if keypoint_2d is not None:
        for joint2d in keypoint_2d:
            x = joint2d['u'] * w
            y = joint2d['v'] * h
            plt.plot(x, y, "o", markersize=3, color="white", alpha=joint2d['confidence'])
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = bbox[0] + bbox[2]
    max_y = bbox[1] + bbox[3]
    plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y])
    plt.show() 
    plt.savefig('swaydemo.jpg')
    return



@util.cache_result_on_disk(f'{paths.CACHE_DIR}/sway.pkl', min_time="2023-06-27T11:30:43")
def make_sway():
    root_sway = f'{paths.DATA_ROOT}/sway'
    joint_names = (
        # 22 smpl joints
        #'hips,lhip,rhip,spin,lkne,rkne,spi1,lank,rank,spi2,ltoe,rtoe,'
        #'neck,lsho,rsho,head,luar,ruar,lelb,relb,lwri,rwri'.split(','))
        'head,lsho,lelb,lwri,rsho,relb,rwri,hips'.split(','))
    edges = (
        'lwri-lelb-luar-ruar-relb-rwri,head-hips')  # ',head-(neck)-hips'
    joint_info = ps3d.JointInfo(joint_names, edges)
    i_relevant_joints = [15, 16, 18, 20, 17, 19, 21, 0]
    frame_step = 5

    def get_examples(phase, pool):
        result = []
        with open(f'{root_sway}/{phase}.txt', "r") as f:
            seq_names = [line.strip() for line in f.readlines()]
        for seq_name in util.progressbar(seq_names):
            seq_path = os.path.join(root_sway, 'sway61769', seq_name)
            intrinsics = np.load(os.path.join(seq_path, "intrinsics.npy"))
            extrinsics = np.load(os.path.join(seq_path, "extrinsics.npy"))
            if np.isnan(extrinsics).any() or np.isnan(intrinsics).any():
                continue
            camera = cameralib.Camera(
                extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                world_up=(0, 1, 0))
            #print(f"Camera R {camera.R}\n t {camera.t}\n intrinsic {camera.intrinsic_matrix}")
            #camera.t *= 1000

            #keypoints = json.load(open(os.path.join(seq_path, "keypts2d.json"), 'r'))['key_points']
            world_pose3d = np.load(os.path.join(seq_path, "wspace_poses3d.npy"))
            if np.isnan(world_pose3d).any():
                continue
            #cam_pose3d = np.load(os.path.join(seq_path, "cspace-poses3d.npy"))
            bbox = np.load(os.path.join(seq_path, "bbox.npy"))
            if np.isnan(bbox).any():
                continue
            n_frames = world_pose3d.shape[0]
            prev_coords = None
            
            for i_frame in range(0, n_frames, frame_step):
                world_coords = world_pose3d[i_frame]
                # print("before", world_coords.shape, world_coords)
                world_coords = world_coords[i_relevant_joints, :]
                # print("after", world_coords.shape, world_coords)                
                if (prev_coords is not None and
                        np.all(np.linalg.norm(world_coords - prev_coords, axis=1) < 100)):
                    continue
                prev_coords = world_coords

                impath = f'sway/sway61769/{seq_name}/images/{i_frame+1:05d}.jpg'
                ex = ps3d.Pose3DExample(impath, world_coords, bbox=bbox[i_frame], camera=camera)
                proj2d = camera.world_to_image(world_coords)
                #print(f'key {proj2d}\nBBox{bbox[i_frame]}')
                #vis(os.path.join(paths.DATA_ROOT, impath), proj2d, bbox[i_frame])
                
                new_image_relpath = impath.replace('sway/sway61679', 'sway_downscaled')
                pool.apply_async(
                    make_efficient_example,
                    (ex, new_image_relpath),
                    callback=result.append)
        return result
            
    with util.BoundedPool(None, 120) as pool:
        train_examples = get_examples('train', pool)
        valid_examples = get_examples('validation', pool)
        test_examples = get_examples('test', pool)

    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ps3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)
