from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import euler
import tensorflow as tf
import tf_context as ctx
import tf_euler
import time


if __name__ == '__main__':
  zk_addr = ctx.get_config('euler', 'zk_addr')
  zk_path = '/euler/{}'.format(ctx.get_app_id())
  shard_num = ctx.get_config('euler', 'shard_num')

  directory = ctx.get_config('euler', 'data_dir')
  shard_idx = ctx.get_task_index() % shard_num

  euler.start(
      directory=directory,
      shard_idx=shard_idx,
      shard_num=shard_num,
      zk_addr=zk_addr,
      zk_path=zk_path,
      module=euler.Module.DEFAULT_MODULE)
  while True:
    time.sleep(1)