import argparse
import json
import os.path
import subprocess
import sys
import numpy as np

import model_train_finetune as kgbnet
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import *
import tf_context as ctx
import runner

def inference(datas):
    sim = kgbnet.inference(datas, ctx.get_config('keep_prob'), ctx.get_config("ps", "instance_num"))
    loss,print_list = kgbnet.sigmoid_loss(sim, ctx.get_config('weight_decay'), ctx.get_config("gamma"))
    auc_op = kgbnet.auc(sim, ctx.get_config('auc_bucket_num'), ctx.get_config('auc_decay_rate'))
    #loss=tf.reduce_mean(sim)
    return loss, auc_op, None, print_list


if __name__ == '__main__':
    # time.sleep(120)
    runner.worker_do(inference)