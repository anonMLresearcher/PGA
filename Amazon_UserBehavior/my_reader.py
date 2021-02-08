from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from datetime import  *
import time
import json
from tf_readers.hooks import *
from tf_readers.file_splits import *
from tf_readers import simple
from tf_readers import rnn
from tensorflow.python.lib.io import file_io
import tf_context as ctx

class MyReader():
  def __init__(self, a, b, config):
    self.batch_size = config.get("batch_size", 1024)

  def set_fields_run_hook(self, fields_run_hook):
    self._fields_run_hook = fields_run_hook
     
  def set_sync_hook(self, sync_hook):
    self._sync_hook = sync_hook
 
  def is_sync_wait(self, sess, global_step):
    if self._sync_hook:
      if not self._sync_hook.is_wait:
        if self._fields_run_hook:
          self._fields_run_hook.start()
      else:
        if self._fields_run_hook:
          self._fields_run_hook.wait_stop()
        sess.run(global_step)
        time.sleep(20)
        return True
    return False
  
  def _get_task_num(self):
    role = ctx.get_task_name()
    return ctx.get_config(role, 'instance_num') or \
           ctx.get_config('extend_role', role, 'instance_num') or 1

  def read(self):
    with ctx.local():
      data_dir = ctx.get_config('euler', 'data_dir') + '/NodeList'
      filenames = file_io.list_directory(data_dir)
      filenames = [data_dir + '/' + filename for filename in filenames]

      import multiprocessing

      parse_fn = lambda row: tf.decode_csv(
          row, [tf.constant(['']), tf.constant([-1], tf.int64)])
      dataset = tf.data.Dataset.list_files(filenames, shuffle=False)
      dataset = dataset.shard(self._get_task_num(), ctx.get_task_index())
      dataset = dataset.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TextLineDataset(x), cycle_length=4))
      dataset = dataset.batch(self.batch_size)
      dataset = dataset.map(parse_fn, num_parallel_calls=multiprocessing.cpu_count())
      dataset = dataset.prefetch(buffer_size=1)
      node_id, source = dataset.make_one_shot_iterator().get_next()
      node_id = tf.expand_dims(node_id, -1)
    return [node_id, source]
   