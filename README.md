# MeTRO Implementation
1. Setup environment in MLP
https://roblox.atlassian.net/wiki/spaces/~ylee/pages/2106262747/MeTRo+training+setup+on+MLP
    * The datasets are organized within /home/jovyan/data/metrabs-processed/
    * The trained models are stored in /home/jovyan/runs/metrabs-exp/$log_dir/model/

2. Training Examples

    1. Trained models are listed in https://docs.google.com/spreadsheets/d/1s_EU91shYbMo9f5R5C5338an3WXiY3TREv5LAt8Gn-0/edit#gid=1046801698

    2. As of 08/14/2023, we have chosen three models that offer the best trade-off, and you can find their respective training commands by `grep command ${ckpt_dir}/log.txt`.
        1. Large model(116M): `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=50000 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=sway_ncnn/256in_ms_w1_up_pv05_abs0_scratch_2d --proc-side=256 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=1.0 --zero-padding=0 --no-geom-aug --occlude-aug-prob=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch`
        2. Medium model(46M): `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=50000 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=sway_ncnn/160in_ms_w1_up_pv05_abs0_scratch --proc-side=160 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=1.0 --zero-padding=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch`
        3. Small model(20M): `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=50000 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=sway_ncnn/112in_ms_w075_up_pv05_abs0_scratch --proc-side=112 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=0.75 --zero-padding=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch`
    
    3. Commands for training METRABS with SimCC heads: 

      1. SimCCSoftArgMax : `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=100 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=simcc/t3 --proc-side=256 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=1.0 --zero-padding=0 --no-geom-aug --occlude-aug-prob=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch --metrabs-simcc-head=SimCCSoftArgMax`

      Note: There is an additional command-line argument `--depth-length` (default value: 128) which determines the number of bins in the final layer of MetrabsSimCCSoftArgMaxHeads

      2. SimCCSoftMax : `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=100 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=simcc/t2 --proc-side=256 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=1.0 --zero-padding=0 --no-geom-aug --occlude-aug-prob=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch --metrabs-simcc-head=SimCCSoftMax`

      3. Vanilla METRABS Head (default): `./main.py --train --dataset=sway --train-on=trainval --seed=1 --training-steps=1000000 --model-class=Metrabs --workers=30 --validate-period=100 --final-transposed-conv=1 --backbone=mobilenetV3Small --logdir=simcc/t1 --proc-side=256 --output-upper-joints --upper-bbox --partial-visibility-prob=0.5 --mobilenet-alpha=1.0 --zero-padding=0 --no-geom-aug --occlude-aug-prob=0 --occlude-aug-prob-2d=0 --dtype=float32 --absloss-factor=0 --data-format=NCHW --input-float32 --init=scratch`

    4. You can manually export a checkpoint into pb format by `ckptno=20001; ./main.py --export-file=model$ckptno --dataset=sway --checkpoint-dir=sway_ncnn/256in_ms_w1_up_pv05_abs0_ckpt --load-path=ckpt-$ckptno --model-class=Metrabs --final-transposed-conv=1 --init=scratch --backbone=mobilenetV3Small --proc-side=256 --mobilenet-alpha=1.0 --output-upper-joints --data-format=NCHW --input-float32 --dtype=float32`
        * Please ensure that the `data format`, `input-float32`, and `dtype` are consistent with the training settings when exporting the model.

3. Visualization
`python src/viz.py model1 .. modelN h36m|input_path output_path`

4. Evaluation
    1. Inference results are stored in the model folder under the filename `predictions_sway.npz` by `metrabs/src$ ./main.py --predict --dataset=sway --checkpoint-dir=sway_ncnn/160in_ms_w1_up_pv05_abs0_scratch --backbone=mobilenetV3Small --init=scratch --mobilenet-alpha=1.0 --model-class=Metrabs --proc-side=160 --output-upper-joints --upper-bbox --upper-bbox-ratio 0.5 0.5 --data-format=NCHW --input-float32`
    2. MPJPE is calculated by `metrabs/src$ python -m eval_scripts.eval_sway --pred-path=/home/jovyan/runs/metrabs-exp/sway_ncnn/160in_ms_w1_up_pv05_abs0_scratch/predictions_sway.npz --root-last`

5. Data preparation
You can incorporate 3D datasets by following these steps:
    1. Integrate a offline dataprocessing function into the `data/datasets3d.py` file. Refer to the example at https://github.rbx.com/ylee/metrabs/pull/10 for guidance
    2. Include the offline data processing function within the `data/sway.py` file.
    3. To evaluate models using the new data, develop the evaluation script within the `eval_scripts/eval_sway.py` file.

6. TF Model conversion to ONNX and NCNN, and bundle for loom-sdk
   You can convert from a TF model with a single command all the way to bundled NCNN:
   ```
   python tools/metrabs_converter.py \
   --output-models-folder ~/git/ave-models/body/ \
   --version 2 1 1 4 \
   --model-path ~/Downloads/160in_ms_w1_up_pv05_abs0_scratch/model \
   --model-kind Metrabs \
   --ave-tracker-path /Users/inavarro/git/ave-tracker/build/client/release \
   --keep-intermediate-files \
   --input-resolution-hw 160 160 \
   --model-input-name input_2
   ```
   This requires having ave-tracker tools installed in your system. You can install ave-tracker tools following the README in https://github.rbx.com/GameEngine/ave-tracker
   In case you would like to convert just to ONNX you can do it without ave-tracker with the following command (avoiding the `--ave-tracker-path` argument):
   ```
   python tools/metrabs_converter.py \
   --output-models-folder ~/git/ave-models/body/ \
   --version 2 1 1 4 \
   --model-path ~/Downloads/160in_ms_w1_up_pv05_abs0_scratch/model \
   --model-kind Metrabs \
   --keep-intermediate-files \
   --input-resolution-hw 160 160 \
   --model-input-name input_2
   ```



# MeTRAbs Absolute 3D Human Pose Estimator

<a href="https://colab.research.google.com/github/isarandi/metrabs/blob/master/metrabs_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metrabs-metric-scale-truncation-robust/3d-human-pose-estimation-on-3d-poses-in-the)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3d-poses-in-the?p=metrabs-metric-scale-truncation-robust)

<p align="center"><img src=img/demo.gif width="60%"></p>
<p align="center"><a href="https://youtu.be/4VFKiiW9RCQ"><img src=img/thumbnail_video_qual.png width="30%"></a>
<a href="https://youtu.be/BemM8-Lx47g"><img src=img/thumbnail_video_conf.png width="30%"></a></p>

This repository contains code for the methods described in the following paper:

**[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227)** <br>
*by István Sárándi, Timm Linder, Kai O. Arras, Bastian Leibe*<br>
IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), Selected Best Works From
Automated Face and Gesture Recognition 2020

## News

* [2021-12-03] Added new backbones, including the ResNet family from ResNet-18 to ResNet-152
* [2021-10-19] Released new best-performing [models](docs/MODELS.md) based on EfficientNetV2 and super fast
  ones using MobileNetV3, simplified [API](docs/API.md), multiple skeleton conventions, support for
  radial/tangential distortion, improved antialiasing, plausibility filtering and other new
  features.
* [2021-10-19] Full codebase migrated to TensorFlow 2 and Keras
* [2020-11-19] Oral presentation at the IEEE Conference on Automatic Face and Gesture Recognition
  (FG'20) ([Talk Video](https://youtu.be/BemM8-Lx47g)
  and [Slides](https://vision.rwth-aachen.de/media/papers/203/slides_metrabs.pdf))
* [2020-11-16] Training and evaluation code now released along with dataset pre-processing scripts!
  Code and models upgraded to Tensorflow 2.
* [2020-10-06] [Journal paper](https://arxiv.org/abs/2007.07227) accepted for publication in the
  IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), Best of FG Special Issue
* [2020-08-23] Short presentation at ECCV2020's 3DPW
  workshop ([slides](https://vision.rwth-aachen.de/media/papers/203/metrabs_3dpw_slides.pdf))
* [2020-08-06] Our method has won
  the **[3DPW Challenge](https://virtualhumans.mpi-inf.mpg.de/3DPW_Challenge/)**

## Inference Code

We release **standalone TensorFlow models** (SavedModel) to allow easy application in downstream
research. After loading the model, you can run inference in a single line of Python **without having
this codebase as a dependency**. Try it in action in
[Google Colab](
https://colab.research.google.com/github/isarandi/metrabs/blob/master/metrabs_demo.ipynb).

### Gist of Usage

```python
import tensorflow as tf

model = tf.saved_model.load('path/to/metrabs_eff2l_y4')
image = tf.image.decode_jpeg(tf.io.read_file('img/test_image_3dpw.jpg'))
pred = model.detect_poses(image)
pred['boxes'], pred['poses2d'], pred['poses3d']
```

NOTE: The models can only be used for **non-commercial** purposes due to the licensing of the used
training datasets.

### Demos

* [```./demo.py```](demo.py) to auto-download the model, predict on a sample image and display the
  result with Matplotlib or [PoseViz](https://github.com/isarandi/poseviz) (if installed).
* [```./demo_webcam.py```](demo_webcam.py) to show webcam inference with the MobileNetV3 backbone (fast but
  lower accuracy, requires [PoseViz](https://github.com/isarandi/poseviz)).
* [```./demo_video_batched.py```](demo_video_batched.py)``` path/to/video.mp4``` to run batched video inference (requires
  [PoseViz](https://github.com/isarandi/poseviz)).

### Documentation

- **[How-to Guide with Examples](docs/INFERENCE_GUIDE.md)**
- **[Full API Reference](docs/API.md)**
- **[Model Zoo with Downloads](docs/MODELS.md)**

### Feature Summary

- **Several skeleton conventions** supported through the keyword argument ```skeleton``` (e.g. COCO,
  SMPL, H36M)
- **Multi-image (batched) and single-image** predictions both supported
- **Advanced, parallelized cropping** logic behind the scenes
    - Anti-aliasing through image pyramid and
      supersampling, [gamma-correct rescaling](http://www.ericbrasseur.org/gamma.html).
    - GPU-accelerated undistortion of pinhole perspective (homography) and radial/tangential lens
      distortions
- Estimates returned  in **3D world space** (when calibration is provided) and **2D pixel space**
- Built-in, configurable **test-time augmentation** (TTA) with rotation, flip and brightness (keyword
  argument ```num_aug``` sets the number of TTA crops per detection)
- Automatic **suppression of implausible poses** and non-max suppression on the 3D pose level (can be turned off)
- **Multiple backbones** with different speed-accuracy trade-off (EfficientNetV2, MobileNetV3)

## Training and Evaluation

See [the docs on training and evaluation](TRAINING_AND_EVAL.md) to perform the experiments presented
in the paper.

## BibTeX

If you find this work useful in your research, please cite it as:

```bibtex
@article{sarandi2021metrabs,
  title={{MeTRAbs:} Metric-Scale Truncation-Robust Heatmaps for Absolute 3{D} Human Pose Estimation},
  author={S\'ar\'andi, Istv\'an and Linder, Timm and Arras, Kai O. and Leibe, Bastian},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2021},
  volume={3},
  number={1},
  pages={16-30},
  doi={10.1109/TBIOM.2020.3037257}
}
```

The above paper is an extended journal version of the FG'2020 conference paper:

```bibtex
@inproceedings{Sarandi20FG,
  title={Metric-Scale Truncation-Robust Heatmaps for 3{D} Human Pose Estimation},
  author={S\'ar\'andi, Istv\'an and Linder, Timm and Arras, Kai O. and Leibe, Bastian},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition},
  pages={677-684},
  year={2020}
}
```

## Contact

Code in this repository was written by [István Sárándi](https://isarandi.github.io) (RWTH Aachen
University) unless indicated otherwise.

Got any questions or feedback? Drop a mail to sarandi@vision.rwth-aachen.de!
