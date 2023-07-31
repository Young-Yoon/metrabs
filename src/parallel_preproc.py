import atexit
import itertools
from options import logger
import os
import queue
import threading

import more_itertools
import numpy as np
import tensorflow as tf

import my_itertools
import tfu
import util


def parallel_map_as_tf_dataset(
        fun, iterable, *, shuffle_before_each_epoch=False,
        extra_args=None, n_workers=10, rng=None, max_unconsumed=256, n_completed_items=0,
        n_total_items=None, roundrobin_sizes=None, use_tfrecord=False):
    """Maps `fun` to each element of `iterable` and wraps the resulting sequence as
    as a TensorFlow Dataset. Elements are processed by parallel workers using `multiprocessing`.

    Args:
        fun: A function that takes an element from seq plus `extra_args` and returns a sequence of
        numpy arrays.
        seq: An iterable holding the inputs.
        shuffle_before_each_epoch: Shuffle the input elements before each epoch. Converts
            `iterable` to a list internally.
        extra_args: extra arguments in addition to an element from `seq`,
            given to `fun` at each call
        n_workers: Number of worker processes for parallelity.

    Returns:
        tf.data.Dataset based on the arrays returned by `fun`.
    """

    extra_args = extra_args or []

    # Automatically determine the output tensor types and shapes by calling the function on
    # the first element
    if not roundrobin_sizes:
        iterable = more_itertools.peekable(iterable)
        first_elem = iterable.peek()
    else:
        iterable[0] = more_itertools.peekable(iterable[0])
        first_elem = iterable[0].peek()

    sample_output = fun(first_elem, *extra_args, rng=np.random.RandomState(0))
    output_signature = tf.nest.map_structure(tf.type_spec_from_value, sample_output)

    if not roundrobin_sizes:
        items = my_itertools.iterate_repeatedly(
            iterable, shuffle_before_each_epoch, util.new_rng(rng))
    else:
        items = my_itertools.roundrobin_iterate_repeatedly(
            iterable, roundrobin_sizes, shuffle_before_each_epoch, rng)

    # If we are restoring from a checkpoint and have already completed some
    # training steps for that checkpoint, then we need to advance the RNG
    # accordingly, to continue exactly where we left off.
    iter_rng = util.new_rng(rng)
    util.advance_rng(iter_rng, n_completed_items)
    items = itertools.islice(items, n_completed_items, n_total_items)

    if n_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 12)
    if n_workers == 0:
        def gen():
            for item in items:
                yield fun(item, *extra_args, util.new_rng(iter_rng))
    else:
        gen = parallel_map_as_generator(
            fun, items, extra_args, n_workers, rng=iter_rng, max_unconsumed=max_unconsumed)

    if use_tfrecord:
        import paths
        import data.datasets3d as ps3d
        filenames = tf.data.Dataset.list_files(f'{paths.CACHE_DIR}/tfrecord/sway_train_*.tfrecord')
        ds = filenames.apply(
            tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=4))

        def parse_fun(raw_record):
            ex = tf.train.Example()
            ex.ParseFromString(raw_record.numpy())
            feature = tfu.tf_example_to_feature(ex)
            new_ex = ps3d.init_from_feature(feature)
            result = fun(new_ex, *extra_args, util.new_rng(iter_rng))
            keys = sorted(result.keys())
            '''
            print('result: ', keys, 'fun: ', fun)
            for k in keys:
                if isinstance(result[k], np.ndarray):
                    print(k, result[k].shape, result[k].dtype, result[k].flatten()[0])
                else:
                    print(k, type(result[k]))
            ---
            cam_loc float32 0.0
            coords2d_true float32 87.22508
            coords3d_true float32 79.01264
            image float32 0.72156864
            image_path <class 'str'>
            intrinsics float32 476.31747
            is_joint_in_fov float32 1.0
            joint_validity_mask bool True
            rot_to_orig_cam float32 1.0
            rot_to_world float32 -0.9978849
            '''
            res = [result[k] for k in keys]
            # print('At parse_fun', [(i, type(v), len(v) if isinstance(v, str) else (v.dtype, v.shape)) for i, v in enumerate(res)])
            # At parse_fun [(0, <class 'numpy.ndarray'>, (dtype('float32'), (3,))), (1, <class 'numpy.ndarray'>, (dtype('float32'), (17, 2))), (2, <class 'numpy.ndarray'>, (dtype('float32'), (17, 3))), (3, <class 'numpy.ndarray'>, (dtype('float32'), (160, 160, 3))), (4, <class 'str'>, 79), (5, <class 'numpy.ndarray'>, (dtype('float32'), (3, 3))), (6, <class 'numpy.ndarray'>, (dtype('float32'), (17,))), (7, <class 'numpy.ndarray'>, (dtype('bool'), (17,))), (8, <class 'numpy.ndarray'>, (dtype('float32'), (3, 3))), (9, <class 'numpy.ndarray'>, (dtype('float32'), (3, 3)))]
            res = [tf.convert_to_tensor(r) for r in res]
            # print('parse_fun:to_tensor', [(i, type(v), len(v) if isinstance(v, str) else (v.dtype, v.shape)) for i, v in enumerate(res)])
            # parse_fun:to_tensor [(0, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([3]))), (1, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([17, 2]))), (2, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([17, 3]))), (3, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([160, 160, 3]))), (4, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.string, TensorShape([]))), (5, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([3, 3]))), (6, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([17]))), (7, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.bool, TensorShape([17]))), (8, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([3, 3]))), (9, <class 'tensorflow.python.framework.ops.EagerTensor'>, (tf.float32, TensorShape([3, 3])))]
            # exit()
            return res
        
        for raw_record in ds.take(1):
            parse_sample = parse_fun(raw_record)
            parse_signature = tf.nest.map_structure(tf.type_spec_from_value, parse_sample)
            # print(parse_signature)  # [TensorSpec(shape=(3,), dtype=tf.float32, name=None), TensorSpec(shape=(17, 2), dtype=tf.float32, name=None), TensorSpec(shape=(17, 3), dtype=tf.float32, name=None), TensorSpec(shape=(160, 160, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None), TensorSpec(shape=(17,), dtype=tf.float32, name=None), TensorSpec(shape=(17,), dtype=tf.bool, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None)]
            parse_shape = [e.get_shape().as_list() for e in parse_sample]
            print(parse_shape)  # [[3], [17, 2], [17, 3], [160, 160, 3], [], [3, 3], [17], [17], [3, 3], [3, 3]]

        tf_parse_fun = lambda x: tf.py_function(parse_fun, inp=[x], Tout=tuple(parse_signature)) #(tf.float32, tf.float32, tf.float32, tf.float32, tf.string, tf.float32, tf.float32, tf.bool, tf.float32, tf.float32)),
        for raw_record in ds.take(1):
            tf_parse_sample = tf_parse_fun(raw_record)
            # print(type(tf_parse_sample), [(type(e), e.shape, e.dtype)for e in tf_parse_sample])
            # <class 'list'> [(<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([3]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([17, 2]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([17, 3]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([160, 160, 3]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([]), tf.string), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([3, 3]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([17]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([17]), tf.bool), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([3, 3]), tf.float32), (<class 'tensorflow.python.framework.ops.EagerTensor'>, TensorShape([3, 3]), tf.float32)]
        #exit()

        def ensure_shape(*args):
            res = []
            for v, s in zip(args, parse_shape):
                v.set_shape(s)
                res.append(v)
            # print(res) # [<tf.Tensor 'args_0:0' shape=(3,) dtype=float32>, <tf.Tensor 'args_1:0' shape=(17, 2) dtype=float32>, <tf.Tensor 'args_2:0' shape=(17, 3) dtype=float32>, <tf.Tensor 'args_3:0' shape=(160, 160, 3) dtype=float32>, <tf.Tensor 'args_4:0' shape=() dtype=string>, <tf.Tensor 'args_5:0' shape=(3, 3) dtype=float32>, <tf.Tensor 'args_6:0' shape=(17,) dtype=float32>, <tf.Tensor 'args_7:0' shape=(17,) dtype=bool>, <tf.Tensor 'args_8:0' shape=(3, 3) dtype=float32>, <tf.Tensor 'args_9:0' shape=(3, 3) dtype=float32>]
            return tuple(res)

        def create_dict(cam_loc, co2d, co3d, image, impath, intrinsics, joint_in, mask, rot_cam, rot_world):
            inps = (cam_loc, co2d, co3d, image, impath, intrinsics, joint_in, mask, rot_cam, rot_world)
            # print('At create_dict', [(type(x), x.shape, x.dtype, x.ref().deref()) for x in inps])
            # At create_dict [(<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.string), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.bool), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32), (<class 'tensorflow.python.framework.ops.Tensor'>, TensorShape(None), tf.float32)]
            return dict(cam_loc=cam_loc,
                    coords2d_true=co2d,
                    coords3d_true=co3d,
                    image=tf.convert_to_tensor(image),
                    image_path=impath,
                    intrinsics=intrinsics,
                    is_joint_in_fov=joint_in,
                    joint_validity_mask=mask,
                    rot_to_orig_cam=rot_cam,
                    rot_to_world=rot_world)
            '''
            return {'cam_loc': tf.convert_to_tensor(cam_loc), 
                    'coords2d_true': tf.convert_to_tensor(co2d), 
                    'coords3d_true': tf.convert_to_tensor(co3d), 
                    'image': tf.convert_to_tensor(image),
                    'image_path': tf.convert_to_tensor(impath),
                    'intrinsics': tf.convert_to_tensor(intrinsics),
                    'is_joint_in_fov': tf.convert_to_tensor(joint_in),
                    'joint_validity_mask': tf.convert_to_tensor(mask),
                    'rot_to_orig_cam': tf.convert_to_tensor(rot_cam),
                    'rot_to_world': tf.convert_to_tensor(rot_world)}'''

        # ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        print(ds)  # <ParallelInterleaveDataset element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)>
        ds = ds.map(tf_parse_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(ds)  # <ParallelMapDataset element_spec=(TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.string, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.bool, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), TensorSpec(shape=<unknown>, dtype=tf.float32, name=None))>
        ds = ds.map(ensure_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(ds)  # <MapDataset element_spec=(TensorSpec(shape=(3,), dtype=tf.float32, name=None), TensorSpec(shape=(17, 2), dtype=tf.float32, name=None), TensorSpec(shape=(17, 3), dtype=tf.float32, name=None), TensorSpec(shape=(160, 160, 3), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None), TensorSpec(shape=(17,), dtype=tf.float32, name=None), TensorSpec(shape=(17,), dtype=tf.bool, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None), TensorSpec(shape=(3, 3), dtype=tf.float32, name=None))>
        ds = ds.map(create_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(ds)  # <MapDataset element_spec={'cam_loc': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'coords2d_true': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'coords3d_true': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'image': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'image_path': TensorSpec(shape=<unknown>, dtype=tf.string, name=None), 'intrinsics': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'is_joint_in_fov': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'joint_validity_mask': TensorSpec(shape=<unknown>, dtype=tf.bool, name=None), 'rot_to_orig_cam': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), 'rot_to_world': TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)}>
        #exit()
    else:
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # Make the cardinality of the dataset known to TF.
    if n_total_items is not None:
        ds = ds.take(n_total_items - n_completed_items)
    return ds


def parallel_map_as_generator(fun, items, extra_args, n_workers, max_unconsumed=256, rng=None):
    semaphore = threading.Semaphore(max_unconsumed)
    q = queue.Queue()
    end_of_sequence_marker = object()
    should_stop = False
    pool = tfu.get_pool(n_workers)

    def producer():
        for i_item, item in enumerate(items):
            if should_stop:
                break
            semaphore.acquire()
            q.put(pool.apply_async(fun, (item, *extra_args, util.new_rng(rng))))

        logger.debug('Putting end-of-seq')
        q.put(end_of_sequence_marker)

    def consumer():
        while (future :=q.get()) is not end_of_sequence_marker:
            value = future.get()
            semaphore.release()
            yield value

    def stop():
        nonlocal should_stop
        should_stop = True
        pool.close()
        pool.terminate()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    atexit.register(stop)

    return consumer
