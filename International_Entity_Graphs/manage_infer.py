import argparse
import json
import os.path
import subprocess
import sys
import numpy as np

import model_infer as kgbnet
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import *
import tf_context as ctx
import my_runner
import tf_euler
from base_runner.utils import tracer_bin

def init_graph():
    zk_addr = ctx.get_config('euler', 'zk_addr')
    zk_path = '/euler/{}'.format(ctx.get_app_id())
    shard_num = ctx.get_config('euler', 'shard_num')
    tf_euler.initialize_graph({
          'mode': 'remote',
          'zk_server': zk_addr,
          'zk_path': zk_path,
          'shard_num': shard_num,
          'num_retries':1
      })

def inference(datas):
    sim = kgbnet.inference(datas, ctx.get_config('keep_prob'), ctx.get_config("ps", "instance_num"))
    # loss = kgbnet.softmax_loss(sim, ctx.get_config('weight_decay'), ctx.get_config("gamma"))
    # auc_op = kgbnet.auc(sim, ctx.get_config('auc_bucket_num'), ctx.get_config('auc_decay_rate'))
    output_dir = ctx.get_config('euler', 'output_dir')
    ctx.add_hook(
          tracer_bin.TracerHook({'format': 'bin', 'output_dir': output_dir}))
    # loss=tf.reduce_mean(sim)
    return tf.reduce_mean(sim),tf.reduce_mean(sim)

if __name__ == '__main__':
    init_graph()
    my_runner.worker_do(inference)