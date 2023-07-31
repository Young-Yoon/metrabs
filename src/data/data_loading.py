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
    with open('load_options.txt', 'a') as f:
        f.write(', '.join([str(e) for e in [ex.image_path, learning_phase,
          FLAGS.proc_side, FLAGS.upper_bbox, FLAGS.crop_mode, 
          FLAGS.partial_visibility_prob, # util.random_partial_subbox()
          FLAGS.geom_aug, FLAGS.shift_aug, 
          FLAGS.occlude_aug_prob, # augmentation.appearance.augment_appearance(im)
          FLAGS.color_aug, # augmentation.color.augment_color(im)
          FLAGS.zero_padding]])+'\n')
    # Get the random number generators for the different augmentations to make it reproducibile
    appearance_rng = util.new_rng(rng)
    background_rng = util.new_rng(rng)
    geom_rng = util.new_rng(rng)
    partial_visi_rng = util.new_rng(rng)
    upperbody_indices = [0, 9, 10, 11, 12, 13, 14, 15, 16]
    output_side = FLAGS.proc_side
    output_imshape = (output_side, output_side)

    # Load and reproject image
    image_path = util.ensure_absolute_path(ex.image_path)
    if ex.image_numpy is None:
        origsize_im = improc.imread_jpeg(image_path)
    else:
        origsize_im = ex.image_numpy
        # print('load:', len(ex.image_numpy))
    h, w, _ = origsize_im.shape

    box = ex.bbox
    
    # resize bbox using keypoints in image coordinate 
    if FLAGS.upper_bbox:
        full_imgcoords = ex.camera.world_to_image(ex.world_coords)
        upper_imgcoords = full_imgcoords[upperbody_indices]
        box = boxlib.expand_with_margin(boxlib.bb_of_points(upper_imgcoords), [20, 80, 20, 20])

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
        x1, y1, w1, h1 = boxlib.intersect(boxlib.expand_to_square(box), np.array([0, 0, w, h]))
#         x1, y1, w1, h1 = box[:4]
        x1_max, y1_max = x1 + w1, y1 + h1
        side = max(w1, h1)
        
        if FLAGS.crop_mode in [0, 1]:    
            im = np.array(origsize_im)[int(y1):int(y1_max), int(x1):int(x1_max)]
            xx = int(x1)
            yy = int(y1)
            hh, ww, _ = im.shape
        elif FLAGS.crop_mode == 2:
            crop_img = np.array(origsize_im)[int(y1):int(y1_max), int(x1):int(x1_max)]
            hh, ww, _ = crop_img.shape
            crop_l = max(hh, ww)
            zp_crop_img = np.zeros([crop_l, crop_l, 3], dtype=crop_img.dtype)
            yy = int((crop_l - hh) / 2)
            xx = int((crop_l - ww) / 2)
            zp_crop_img[yy: yy + hh, xx: xx + ww] = crop_img
            im = zp_crop_img
            hh, ww = crop_l, crop_l
            xx = int(x1) - xx
            yy = int(y1) - yy
        elif FLAGS.crop_mode == 3:
            if w1 < h1:
                w3, h3 = w1, w1
                x3, y3 = x1, y1 + (h1 - w1) * 0.1
            else:
                w3, h3 = h1, h1  
                x3, y3 = x1 + (w1 - h1) * 0.5, y1
            x3_max = x3 + w3
            y3_max = y3 + h3
            side = max(w3, h3)
            im = np.array(origsize_im)[int(y3):int(y3_max), int(x3):int(x3_max)]
            xx = int(x3)
            yy = int(y3)
            hh, ww, _ = im.shape
        else:
            print("Error. Unsupported bbox squarization mode. Only 0, 1, 2, 3 are supported!")
            exit()
                    
        if im.shape[0] >= 40 and im.shape[1] >= 40:            
            im = cv2r.resize(im, dsize=(output_imshape[1], output_imshape[0]), interpolation=cv2.INTER_AREA, dst=None)
            resize_factor = side / output_side        
            cam = ex.camera.copy()
            cam.intrinsic_matrix[:2, 2] -= np.array([xx, yy])
            cam.intrinsic_matrix[0] /= (ww / output_side) 
            cam.intrinsic_matrix[1] /= (hh / output_side) 
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

    # Add zero padding on left or right
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

    if FLAGS.save_image_from_loader:
        import matplotlib.pyplot as plt
        plt.imshow(im)
        for coord in imcoords[upperbody_indices]:
            plt.plot(coord[0], coord[1], 'o', color="orange")
        import os
        os.makedirs("tmp", exist_ok=True)
        plt.savefig("tmp/{}.jpg".format(np.random.randint(0, 1000)))
        plt.clf()
    
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
