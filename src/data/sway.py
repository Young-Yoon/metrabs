import os.path
import numpy as np
import itertools
import cameralib
import data.datasets3d as ps3d
import paths
import util
from data.preproc_for_efficiency import make_efficient_example
from data.utils import visualize_sample




def get_seq_info(phase, root_sway, use_kd):
    if use_kd:
        frame_step = 5 if phase in {'train'} else 60
        with open(f'{root_sway}/dataset61769.txt', "r") as f:
            seq_names = [line.strip() for line in f.readlines()]
        parts = [0, 55561, 58648, 61734]   # [12000//100*p for p in (0, 90, 95, 100)]  # sway12k
        pid = {'train':0, 'validation':1, 'test':2}
        seq_names = seq_names[parts[pid[phase]]:parts[pid[phase]+1]]
        seq_folders = ['sway61769']
        if phase in {'test'}:
            seq_folders += ['sway_test_variants/'+v for v in ('landscape', 'portrait', 'tight')]
    else:
        frame_step = 30 if phase in {'train'} else 64
        with open(f'{root_sway}/{phase}.txt', "r") as f:
            seq_names = [line.strip() for line in f.readlines()]
        if phase in {'train'}:
            seq_names = seq_names[70:]
        if phase in {'test'}:
            seq_folders = ['sway61769'] + ['sway_test_variants/'+v for v in ['landscape', 'portrait', 'tight']]
            print(seq_folders)
        else:
            seq_folders = ['sway61769']
    return seq_names, seq_folders, frame_step


def load_seq_param(seq_dir, seq_name, root_sway, use_kd):
    seq_path = os.path.join(root_sway, seq_dir, seq_name)
    if use_kd:
        fi, fe, fw, fb = (os.path.join(seq_path, "metrab_annotations", f) for f in ["intrinsics.npy", "extrinsics.npy", "poses3d.npy", "bbox.npy"])
        if not os.path.exists(fi):
            return False, None
    else:
        fi, fe, fw, fb = (os.path.join(seq_path, f) for f in ["intrinsics.npy", "extrinsics.npy", "wspace_poses3d.npy", "bbox.npy"])

    intrinsics, extrinsics = np.load(fi, allow_pickle=True), np.load(fe, allow_pickle=True)
    #print(intrinsics, extrinsics)
    #if np.isnan(extrinsics).any() or np.isnan(intrinsics).any():
    #    return False, None
    if len(intrinsics.shape) == 2:
        camera = cameralib.Camera(
            extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
            world_up=(0, 1, 0))
    else:
        camera = (intrinsics.item(), extrinsics)

    if use_kd:
        world_pose3d = np.load(fw, allow_pickle=True).item()
        bbox = np.load(fb, allow_pickle=True).item()
        n_frames = int(list(world_pose3d.keys())[-1])
    else:
        world_pose3d = np.load(fw)
        bbox = np.load(fb)
        if np.isnan(world_pose3d).any() or np.isnan(bbox).any():
            return False, None
        n_frames = world_pose3d.shape[0]
    return True, (camera, world_pose3d, bbox, n_frames)


def get_examples(phase, pool, use_kd=True, visualize=False):
    result = []
    if use_kd:
        # From 'pelv,rhip,rkne,rank,lhip,lkne,lank,spin,neck,head,htop,lsho,lelb,lwri,rsho,relb,rwri’ --> root-last
        i_relevant_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0]
    else:
        i_relevant_joints = [2, 5, 8, 1, 4, 7, 9, 12, 15, 15, 16, 18, 20, 17, 19, 21, 0]    
    root_sway = f'{paths.DATA_ROOT}/sway'
    seq_names, seq_folders, frame_step = get_seq_info(phase, root_sway, use_kd)

    for seq_dir, seq_name in util.progressbar(itertools.product(seq_folders, seq_names)):
        load_success, params = load_seq_param(seq_dir, seq_name, root_sway, use_kd)
        if not load_success:
            continue
        camera, world_pose3d, bbox, n_frames = params
        if isinstance(camera, cameralib.Camera):
            fixedCam = True
        else:
            intrinsics, extrinsics = camera
            fixedCam = False

        prev_coords = None
        for i_frame in range(0, n_frames, frame_step):
            if use_kd:
                fr_idx = f'{i_frame+1:05d}'
                if not fixedCam:
                    if fr_idx not in intrinsics.keys():
                        continue
                    #print(type(intrinsics), type(extrinsics), seq_dir, seq_name, i_frame, intrinsics, extrinsics)
                    camera = cameralib.Camera(
                        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics[fr_idx],
                        world_up=(0, 1, 0))                    
                if fr_idx not in world_pose3d.keys():
                    continue
                world_coords = world_pose3d[fr_idx]
                world_coords = world_coords[i_relevant_joints, :]
                bbox_fr = bbox[fr_idx]
            else:
                if not fixedCam:
                    camera = cameralib.Camera(
                        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics[i_frame],
                        world_up=(0, 1, 0))
                world_coords = world_pose3d[i_frame]
                world_coords = world_coords[i_relevant_joints, :]
                bbox_fr = bbox[i_frame]

            if (phase == 'train' and prev_coords is not None and
                    np.all(np.linalg.norm(world_coords - prev_coords, axis=1) < 100)):
                continue
            prev_coords = world_coords

            impath = f'sway/{seq_dir}/{seq_name}/images/{i_frame+1:05d}.jpg'
            ex = ps3d.Pose3DExample(impath, world_coords, bbox=bbox_fr, camera=camera)

            if visualize:
                visualize_sample(os.path.join(paths.DATA_ROOT, impath), bbox=bbox[i_frame])

#                 new_image_relpath = impath.replace('sway/sway61769', 'sway_downscaled')
#                 pool.apply_async(
#                     make_efficient_example,
#                     (ex, new_image_relpath),
#                     callback=result.append)
            result.append(ex)
    return result


#'sway4test.pkl': include sway_test_variants
#'sway_kd12k.pkl': sway12k annotated by the pretrained metrabs
@util.cache_result_on_disk(f'{paths.CACHE_DIR}/sway_kd.pkl', min_time="2023-06-27T11:30:43")
def make_sway():
    joint_names = (
        'rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,'
        'lsho,lelb,lwri,rsho,relb,rwri,pelv'.split(','))

#     joint_names = (
#         # 22 smpl joints
#         'hips,lhip,rhip,spin,lkne,rkne,spi1,lank,rank,spi2,ltoe,rtoe,'
#         'neck,lsho,rsho,head,luar,ruar,lelb,relb,lwri,rwri'.split(','))
#         'head,lsho,lelb,lwri,rsho,relb,rwri,hips'.split(','))

    edges = (
        'htop-head-neck-lsho-lelb-lwri,neck-rsho-relb-rwri,'
        'neck-tors-pelv-lhip-lkne-lank,pelv-rhip-rkne-rank')

    joint_info = ps3d.JointInfo(joint_names, edges)

    with util.BoundedPool(None, 120) as pool:
        train_examples = get_examples('train', pool)
        valid_examples = get_examples('validation', pool)
        test_examples = get_examples('test', pool)

    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ps3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)
