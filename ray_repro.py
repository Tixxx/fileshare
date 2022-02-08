
import sys
sys.path.append('.')  # adds local folder
import math
import horovod.tensorflow.keras as hvd
import horovod.spark.keras as hvd_keras
import functools
from functools import partial
import tensorflow as tf

import horovod.tensorflow.keras as hvd
import os
import glob
import tensorflow.keras.backend as K
import ray

import argparse

class LocalRayContext():
    def __init__(self):
        # Start the Ray cluster or attach to an existing Ray cluster
        # will generate such error if shutdown before new ray init.
        # (pid=raylet) [2021-02-02 04:00:06,190 E 9518 9518] process.cc:498:
        # Failed to kill process 9577 with error system:3: No such process
        # ray.shutdown()
        # ray.init()
        # therefore instead we use ignore_reinit_error to avoid shutdown
        try:
            ray.init(address='auto', ignore_reinit_error=True)
        except Exception as e:
            print(f'LocalRayContext __init__ ray.init address=auto failed with error: {e}')
            try:
                ray.init(ignore_reinit_error=True)
            except Exception as e2:
                print(f'LocalRayContext __init__ ray.init failed again with error: {e2}')

    def run(self, fn, args=[], kwargs={}, env=None):

        @ray.remote(max_calls=1)
        def run_ray_remote(func):
            return func(None)

        ans = [run_ray_remote.remote(lambda w: fn(*args, **kwargs))]
        return ray.get(ans)

    def shutdown(self):
        ray.shutdown()


class LocalRayDLContext(LocalRayContext):
    
    def __init__(self, num_workers=1, optimizer='horovod'):
        super().__init__()
        if optimizer != 'horovod':
            raise Exception(
                'At the moment, synchronous SGD based on horovod '
                'is the only optimizer supported'
            )
        from horovod.ray import RayExecutor
        self.num_workers, self.cpus_per_worker = num_workers, 1
        self.ray_executor = RayExecutor(
            RayExecutor.create_settings(timeout_s=30),
            num_workers=self.num_workers,
            cpus_per_worker=self.cpus_per_worker,
            use_gpu=False,
            gpus_per_worker=0,
        )
        self.ray_executor.start()
        print(
            f'LocalRayDLContext initialized with 1 host and {self.num_workers} slot(s)')

    def num_processes(self):
        return self.num_workers * self.cpus_per_worker

    def run(self, fn, args=[], kwargs={}, env=None):
        print(f'env is {env}, which will not be used in local ray context')
        return self.ray_executor.run(fn, args, kwargs)

    def shutdown(self):
        self.ray_executor.shutdown()
        super().shutdown()
        print('local ray context shutdown')

import horovod.spark.keras.util as keras_util

worker_count=4
def remote_trainer():
    def keras_fn():
        def fn():
            import tensorflow.keras as tf_keras
            return tf_keras
        return fn
    def horovod_fn():
        def fn():
            import horovod.tensorflow.keras as hvd
            return hvd
        return fn
    pin_gpu = hvd_keras.remote._pin_gpu_fn()

    get_keras = keras_fn()
    get_horovod = horovod_fn()
    def train_function(input):

        k = get_keras()
        float_type = tf.keras.backend.floatx()
        #float_type = K.floatx() -> this works

        print(f"horovod.spark.keras.util: using keras in tensorflow=={tf.__version__}")
        print(float_type)
        hvd = get_horovod()
        hvd.init()
        pin_gpu(hvd, tf, k)
        y_true = tf.random.uniform([2, 3])
        y_pred = tf.random.uniform([2, 3])
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        print(ce)
        return 0
    return train_function


################Use ray################
def get_local_ray_ctx():
    print("use not ray client based horovod ray executor")
    return LocalRayDLContext(worker_count)

backend = get_local_ray_ctx()

trainer = remote_trainer()

handle = backend.run(trainer, args=([2]))

print("######finish training")