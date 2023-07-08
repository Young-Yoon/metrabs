"""Functions for loading learning examples from disk and numpy arrays into tensors.
Augmentations are also called from here.
"""
import re

import cv2
import numpy as np

import augmentation.appearance
import augmentation.background
import augmentation.voc_loader
import boxlib
import cameralib
import improc
import tfu
import util
import cv2
import cv2r
from options import FLAGS
from tfu import TRAIN

    
    
def full_bbox(ex, joint_info, learning_phase, output_side, output_imshape, origsize_im, center_point, geom_rng):
    box = ex.bbox
    # The homographic reprojection of a rectangle (bounding box) will not be another rectangle
    # Hence, instead we transform the side midpoints of the short sides of the box and
    # determine an appropriate zoom factor by taking the projected distance of these two points
    # and scaling that to the desired output image side length.
    if box[2] < box[3]:
        # Tall box: take midpoints of top and bottom sides
        delta_y = np.array([0, box[3] / 2])
        sidepoints = center_point + np.stack([-delta_y, delta_y])
    else:
        # Wide box: take midpoints of left and right sides
        delta_x = np.array([box[2] / 2, 0])
        sidepoints = center_point + np.stack([-delta_x, delta_x])

    cam = ex.camera.copy()
    # cam.turn_towards(target_image_point=center_point)
    cam.undistort()
    cam.square_pixels()
    cam_sidepoints = cameralib.reproject_image_points(sidepoints, ex.camera, cam)
    crop_side = np.linalg.norm(cam_sidepoints[0] - cam_sidepoints[1])
    cam.zoom(output_side / crop_side)
    cam.center_principal_point(output_imshape)

    if FLAGS.geom_aug and (learning_phase == TRAIN or FLAGS.test_aug):
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        zoom = geom_rng.uniform(1 - s1, 1 + s2)
        cam.zoom(zoom)
        r = np.deg2rad(FLAGS.rot_aug)
        cam.rotate(roll=geom_rng.uniform(-r, r))

    world_coords = ex.univ_coords if FLAGS.universal_skeleton else ex.world_coords
    metric_world_coords = ex.world_coords

    if learning_phase == TRAIN and geom_rng.rand() < 0.5:
        cam.horizontal_flip()
        # Must reorder the joints due to left and right flip
        camcoords = cam.world_to_camera(world_coords)[joint_info.mirror_mapping]
        metric_world_coords = metric_world_coords[joint_info.mirror_mapping]
    else:
        camcoords = cam.world_to_camera(world_coords)
    imcoords = cam.world_to_image(metric_world_coords)
    interp_str = (FLAGS.image_interpolation_train
                  if learning_phase == TRAIN else FLAGS.image_interpolation_test)
    antialias = (FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test)
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    im = cameralib.reproject_image(
        origsize_im, ex.camera, cam, output_imshape, antialias_factor=antialias, interp=interp)
    
    return cam, camcoords, imcoords, im
    


def load_and_transform3d(ex, joint_info, learning_phase, rng):
    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    output_side = FLAGS.proc_side
    output_imshape = (output_side, output_side)

    # Load and reproject image
    image_path = util.ensure_absolute_path(ex.image_path)
    origsize_im = improc.imread_jpeg(image_path)
    h, w, _ = origsize_im.shape

    box = ex.bbox
    
    
    # resize bbox using keypoints in image coordinate 
    if FLAGS.upper_bbox:
        full_imgcoords = ex.camera.world_to_image(ex.world_coords)
        upper_imgcoords = full_imgcoords[9:]
        min_x, min_y = 999999, 999999
        max_x, max_y = -999999, -999999
        for coord in upper_imgcoords:
            xx, yy = coord
            min_x = min(min_x, xx)
            min_y = min(min_y, yy)
            max_x = max(max_x, xx)
            max_y = max(max_y, yy)
        min_x = max(min_x - 20, 0)
        min_y = max(min_y - 80, 0)
        max_x = min(max_x + 20, box[0] + box[2])
        max_y = min(max_y + 20, box[1] + box[3])
        box = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
    
    
    partial_visi_prob = FLAGS.partial_visibility_prob
    use_partial_visi_aug = (
            (learning_phase == TRAIN or FLAGS.test_aug) and
            partial_visi_rng.rand() < partial_visi_prob)
    
    if use_partial_visi_aug:
        box = util.random_partial_subbox(boxlib.expand_to_square(box), partial_visi_rng)

    # Geometric transformation and augmentation
    crop_side = np.max(box[2:])
    center_point = boxlib.center(box)
    
    if ((learning_phase == TRAIN and FLAGS.geom_aug) or
            (learning_phase != TRAIN and FLAGS.test_aug and FLAGS.geom_aug)):
        center_point += util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * crop_side
        
    if FLAGS.upper_bbox:
        side = int(max(box[2], box[3]))
        side = min(h, w, side)

        min_x = int(center_point[0] - side / 2)
        min_x = max(min_x, 0)
        min_x = min(min_x, w - side)
        
        min_y = int(center_point[1] - side / 2)
        min_y = max(min_y, 0)
        min_y = min(min_y, h - side) 
        
        max_x = min_x + side
        max_y = min_y + side
        
        im = origsize_im[min_y: max_y, min_x: max_x]
        
#         if FLAGS.zero_padding_bbox:
#             print("need to add")
            
#         if FLAGS.crop_long_bbox:
#             print("need to add")
                    
        if im.shape[0] >= 40 and im.shape[1] >= 40:            
            im = cv2r.resize(im, dsize=(output_imshape[1], output_imshape[0]), interpolation=cv2.INTER_AREA, dst=None)
            resize_factor = side / output_side        
            cam = ex.camera.copy()
            cam.intrinsic_matrix[:2, 2] -= np.array([min_x, min_y])
            cam.intrinsic_matrix[:2] /=  resize_factor
            metric_world_coords = ex.world_coords
            camcoords = cam.world_to_camera(metric_world_coords)
            imcoords = cam.world_to_image(metric_world_coords)
        else:
            cam, camcoords, imcoords, im = full_bbox(ex, joint_info, learning_phase, output_side, output_imshape, origsize_im, center_point, geom_rng)
            
    else:       
        cam, camcoords, imcoords, im = full_bbox(ex, joint_info, learning_phase, output_side, output_imshape, origsize_im, center_point, geom_rng)

    # Occlusion and color augmentation
    im = augmentation.appearance.augment_appearance(
        im, learning_phase, FLAGS.occlude_aug_prob, appearance_rng)

    ### Add zero padding on left or right
    if FLAGS.zero_padding > 0:
        crop_h, crop_w, _ = im.shape
        pad_ratio = np.random.uniform(0, FLAGS.zero_padding)
        if np.random.uniform() > 0.5: # left-right padding
            side = int(crop_w * pad_ratio / 2)
            im[:, :side] = 0
            im[:, -side:] = 0
        else:
            side = int(crop_h * pad_ratio / 2)
            im[:side, :] = 0
            im[-side:, :] = 0

    
    im = tfu.nhwc_to_std(im)
    im = improc.normalize01(im)

    # Joints with NaN coordinates are invalid
    is_joint_in_fov = ~np.logical_or(
        np.any(imcoords < 0, axis=-1), np.any(imcoords >= FLAGS.proc_side, axis=-1))
    joint_validity_mask = ~np.any(np.isnan(camcoords), axis=-1)

    rot_to_orig_cam = ex.camera.R @ cam.R.T
    rot_to_world = cam.R.T
    
    
    return dict(
        image=im,
        intrinsics=np.float32(cam.intrinsic_matrix),
        image_path=ex.image_path,
        coords3d_true=np.nan_to_num(camcoords).astype(np.float32),
        coords2d_true=np.nan_to_num(imcoords).astype(np.float32),
        rot_to_orig_cam=rot_to_orig_cam.astype(np.float32),
        rot_to_world=rot_to_world.astype(np.float32),
        cam_loc=cam.t.astype(np.float32),
        joint_validity_mask=joint_validity_mask,
        is_joint_in_fov=np.float32(is_joint_in_fov))



def load_and_transform2d(ex, joint_info, learning_phase, rng):
    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)

    # Load the image
    image_path = util.ensure_absolute_path(ex.image_path)
    im_from_file = improc.imread_jpeg(image_path)

    # Determine bounding box
    bbox = ex.bbox
    if learning_phase == TRAIN and partial_visi_rng.rand() < FLAGS.partial_visibility_prob:
        bbox = util.random_partial_subbox(boxlib.expand_to_square(bbox), partial_visi_rng)

    crop_side = np.max(bbox)
    center_point = boxlib.center(bbox)
    orig_cam = cameralib.Camera.create2D(im_from_file.shape)
    cam = orig_cam.copy()
    cam.zoom(FLAGS.proc_side / crop_side)

    if FLAGS.geom_aug:
        center_point += util.random_uniform_disc(geom_rng) * FLAGS.shift_aug / 100 * crop_side
        s1 = FLAGS.scale_aug_down / 100
        s2 = FLAGS.scale_aug_up / 100
        cam.zoom(geom_rng.uniform(1 - s1, 1 + s2))
        r = np.deg2rad(FLAGS.rot_aug)
        cam.rotate(roll=geom_rng.uniform(-r, r))

    if FLAGS.geom_aug and geom_rng.rand() < 0.5:
        cam.horizontal_flip()
        # Must also permute the joints to exchange e.g. left wrist and right wrist!
        imcoords = ex.coords[joint_info.mirror_mapping]
    else:
        imcoords = ex.coords

    new_center_point = cameralib.reproject_image_points(center_point, orig_cam, cam)
    cam.shift_to_center(new_center_point, (FLAGS.proc_side, FLAGS.proc_side))

    is_annotation_invalid = (np.nan_to_num(imcoords[:, 1]) > im_from_file.shape[0] * 0.95)
    imcoords[is_annotation_invalid] = np.nan
    imcoords = cameralib.reproject_image_points(imcoords, orig_cam, cam)

    interp_str = (FLAGS.image_interpolation_train
                  if learning_phase == TRAIN else FLAGS.image_interpolation_test)
    antialias = (FLAGS.antialias_train if learning_phase == TRAIN else FLAGS.antialias_test)
    interp = getattr(cv2, 'INTER_' + interp_str.upper())
    im = cameralib.reproject_image(
        im_from_file, orig_cam, cam, (FLAGS.proc_side, FLAGS.proc_side),
        antialias_factor=antialias, interp=interp)
    im = augmentation.appearance.augment_appearance(
        im, learning_phase, FLAGS.occlude_aug_prob_2d, appearance_rng)
    im = tfu.nhwc_to_std(im)
    im = improc.normalize01(im)

    backward_matrix = cameralib.get_affine(cam, orig_cam)

    joint_validity_mask = ~np.any(np.isnan(imcoords), axis=1)
    with np.errstate(invalid='ignore'):
        is_joint_in_fov = ~np.logical_or(np.any(imcoords < 0, axis=-1),
                                         np.any(imcoords >= FLAGS.proc_side, axis=-1))
    # We must eliminate NaNs because some TensorFlow ops can't deal with any NaNs touching them,
    # even if they would not influence the result. Therefore we use a separate "joint_validity_mask"
    # to indicate which joint coords are valid.
    imcoords = np.nan_to_num(imcoords)

    return dict(
        image_2d=np.float32(im),
        intrinsics_2d=np.float32(cam.intrinsic_matrix),
        image_path_2d=ex.image_path,
        coords2d_true_2d=np.float32(imcoords),
        joint_validity_mask_2d=joint_validity_mask,
        backward_matrix=np.float32(backward_matrix),
        is_joint_in_fov_2d=is_joint_in_fov)
