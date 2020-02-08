# Copyright 2020 Graphcore Ltd.
import os
import numpy as np

from tensorflow.python.framework import dtypes

from tensorflow.python import ipu
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ipu.scopes import ipu_scope

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.debugging.set_log_device_placement(True)

cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

size = 6

with tf.device("cpu"):
    x_cpu = tf.placeholder(np.float32, [size], name="x_cpu_ph")
    y_cpu = tf.placeholder(np.float32, [size], name="y_cpu_ph")


def my_net(x, y):
    outputs = {
        "output_types": [tf.float32],
        "output_shapes": [tf.TensorShape([4])]
    }

    base_path = os.getcwd()
    lib_path = os.path.join(base_path, "cpu_op.so")
 
    print("Loading " + lib_path)
    return ipu.custom_ops.cpu_user_operation([x, y],
                                              lib_path,
                                              outs=outputs,
                                              name='cpuCallbackOp',
                                              op_name='cpuCallback')


with ipu_scope("/device:IPU:0"):
    #x_ipu = array_ops.placeholder(tf.float32, shape=4)
    #y_ipu = array_ops.placeholder(tf.float32, shape=4)
    xla_result = ipu.ipu_compiler.compile(my_net, inputs=[x_cpu, y_cpu])

with tf.Session() as sess:
    
    
    sess.run(variables.global_variables_initializer())
    a = np.ones([size])
    b = np.ones([size])

    result = sess.run(xla_result, feed_dict = {x_cpu: a, y_cpu: b})

# Show result from the IPU:
print("IPU:", result[0])

# Same calculation on host for comparison:
print("numpy:", a + b)
