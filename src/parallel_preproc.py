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

    if False:  # and use_tfrecord:
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
            return (result[k] for k in keys)

        def create_dict(cam_loc, co2d, co3d, image, impath, intrinsics, joint_in, mask, rot_cam, rot_world):
            return {'cam_loc': cam_loc,
                    'coords2d_true': co2d,
                    'coords3d_true': co3d,
                    'image': image,
                    'image_path': impath,
                    'intrinsics': intrinsics,
                    'is_joint_in_fov': joint_in,
                    'joint_validity_mask': mask,
                    'rot_to_orig_cam': rot_cam,
                    'rot_to_world': rot_world}
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

        ds.map(lambda x: tf.py_function(parse_fun, inp=[x], 
            Tout=(tf.float32, tf.float32, tf.float32, tf.string, tf.string, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).map(create_dict)
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
