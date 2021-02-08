import tensorflow as tf
import tf_context as ctx
from tf_context import hooks
import tf_readers
import my_reader
from base_runner import utils

# inference_fn: function(datas)
# create_optimizer_fn: function(props, global_step)
def worker_do(inference_fn=None, create_optimizer_fn=None):
  if 'train' in ctx.get_config("job_type"):
    train_default(inference_fn, create_optimizer_fn)
  elif 'auc' in ctx.get_config("job_type"):
    auc_default(inference_fn)
  elif 'model_convert' in ctx.get_config("job_type"):
    model_convert_default(inference_fn)
  elif 'model_init' in ctx.get_config("job_type"):
    import tf_model_init
    tf_model_init.model_init_plus(ctx.get_task_index(), ctx.get_config("checkpoint"), ctx.get_config("model_init"))  

def read_data():
  with ctx.local():
    params = [ctx.get_task_index(),
              ctx.get_config("worker", "instance_num"),
              ctx.get_config("reader")]
    if ctx.get_config("reader", "reader_type") == "rnn":
      reader = tf_readers.RnnReader(*params)
    elif ctx.get_config("reader", "reader_type") == "streaming_rnn":
      reader = tf_readers.RnnStreamingReader(*params)
    elif ctx.get_config("reader", "reader_type") == "dolphin":
      reader = tf_readers.SimpleReader(*params)
    elif ctx.get_config("reader", "reader_type") == "steaming_dolphin":
      reader = tf_readers.SimpleStreamingReader(*params)
    elif ctx.get_config("reader", "reader_type") == "my_reader":
      reader = my_reader.MyReader(*params)
    else:
      raise ValueError("config not contains (rnn_reader, streaming_rnn_reader, simple_reader, streaming_simple_reader)")
    return reader.read(), reader

########### train
def train_default(inference_fn=None, create_optimizer_fn=None):
  
  # read
  datas, reader = read_data()
  # graph
  with ctx.graph():
    if ctx.get_config("ps_type") == "ps_native":
      auc_vars = utils.auc_vars.get_native_auc_variables()
    elif ctx.get_config("ps_type") == "ps_plus":
      auc_vars = utils.auc_vars.get_plus_auc_variables()
      finish_rate = ctx.get_config("min_finish_worker_rate")
      if finish_rate is None:
        finish_rate = 90
      ctx.add_hook(utils.worker_finish.ReportFinishHook(ctx.get_task_index()))
      ctx.add_hook(utils.worker_finish.ScheduleFinishHook(ctx.get_config("worker", "instance_num"), finish_rate))      
      
    global_step = tf.contrib.framework.get_or_create_global_step()
    # inference
    ret = inference_fn(datas)
    loss = None
    auc = None
    feed_dict = None
    if len(ret) > 0:
      loss = ret[0]
    if len(ret) > 1:
      auc = ret[1]
    if len(ret) > 2:
      feed_dict = ret[2]
    tf.summary.scalar('loss', loss)
    if auc is not None:
      tf.summary.scalar('auc', auc)
    
    run_ops = loss
    # optimizer
    if ctx.is_chief() and ctx.get_config('graph_tag') is not None:
      import graph_tag
      tag_files = ctx.get_config('graph_tag').split(",")
      for i in range(len(tag_files)):
        graph_tag.write_tag(tag_files[i], i)
    if ctx.get_config("optimizers"):
      opt = utils.optimizers.get_optimizers(ctx.get_config("optimizers"), 
                                            global_step, 
                                            create_optimizer_fn=create_optimizer_fn)
      run_ops = opt.minimize(loss)
      if ctx.get_config("use_batch_norm") is not None:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          run_ops = tf.group(run_ops)

    # send gstep, qps, etc. to swift-queue
    swift_writer = utils.summary.generateWriter(ctx.get_config())
    ctx.add_hook(hooks.SwiftSendHook({"loss": loss, "auc": auc},
                                    log_steps=ctx.get_config("log_steps"),
                                    batch_size=reader.batch_size,
                                    optype='train',
                                    job_conf=ctx.get_config(),
                                    summary_writer=swift_writer))
    # log variables
    ctx.add_hook(hooks.LoggerHook({"loss": loss, "auc": auc},
                                  log_steps=ctx.get_config("log_steps"),
                                  batch_size=reader.batch_size,
                                  job_type="train"))
    if ctx.get_config("feature_expire"):
      print("Adding expire hooks")
      utils.feature_expire.add_expire_hooks(ctx, ctx.get_config("feature_expire"), datas)
      
    # summary
    if ctx.get_config("summary") and ctx.get_task_index() == 0:
      ctx.add_hook(hooks.SummaryHook(ctx.get_config("summary"), summary_writer=swift_writer))
    # reader hooks
    ctx.add_hook(reader.get_hooks())
    # save auc hook
    if ctx.get_config("auc"):
      ctx.add_hook(utils.save_auc.SaveAucHook(auc, global_step,
                                              run_interval=ctx.get_config("log_steps")))
    #  auc tracer
    if ctx.get_config("tracer.bin"):
      ctx.add_hook(utils.tracer_bin.TracerHook(ctx.get_config("tracer.bin"), datas))
    # csv tracer
    if ctx.get_config("trace"):
      ctx.add_hook(utils.tracer_csv.TracerHook(ctx.get_config(), global_step))
    # profiler
    if ctx.get_config("profiler"):
      ctx.add_hook(tf.train.ProfilerHook(save_steps=ctx.get_config("profiler", "save_steps"), output_dir=ctx.get_config("profiler", "output_dir")))
    # create session and run
    with ctx.session() as sess:
      while not sess.should_stop():
        if not reader.is_sync_wait(sess, global_step):
          if feed_dict is not None:
            feed = {k: (v() if callable(v) else v) for k, v in feed_dict.items()}
          else:
            feed = None
          sess.run(run_ops, feed_dict=feed)

########### auc
def auc_default(inference_fn=None):
  
  # read
  datas, reader = read_data()
  # graph
  with ctx.graph():
    global_step = tf.contrib.framework.get_or_create_global_step()
    # inference
    ret = inference_fn(datas)
    loss = None
    auc = None
    feed_dict = None
    if len(ret) > 0:
      loss = ret[0]
    if len(ret) > 1:
      auc = ret[1]
    if len(ret) > 2:
      feed_dict = ret[2]

    zero_ops = None
    if ctx.get_config("ps_type") == "ps_native":
      auc_vars = utils.auc_vars.get_native_auc_variables()
    elif ctx.get_config("ps_type") == "ps_plus":
      auc_vars = utils.auc_vars.get_plus_auc_variables()
      zero_ops = utils.auc_vars.get_plus_zero_ops(ctx.get_config('auc_bucket_num'))
      finish_rate = ctx.get_config("min_finish_worker_rate")
      if finish_rate is None:
        finish_rate = 90
      ctx.add_hook(utils.worker_finish.ReportFinishHook(ctx.get_task_index()))
      ctx.add_hook(utils.worker_finish.ScheduleFinishHook(ctx.get_config("worker", "instance_num"), finish_rate))
    
    # reader hooks
    # ctx.add_hook(reader.get_hooks())

    # send gstep, qps, etc. to swift-queue
    swift_writer = utils.summary.generateWriter(ctx.get_config())
    ctx.add_hook(hooks.SwiftSendHook({"loss": loss, "auc": auc},
                                    log_steps=ctx.get_config("log_steps"),
                                    batch_size=reader.batch_size,
                                    optype='auc_op',
                                    job_conf=ctx.get_config(),
                                    summary_writer=swift_writer))
    # log print
    ctx.add_hook(hooks.LoggerHook({"Loss": loss, "Auc": auc}, 
                                    log_steps=ctx.get_config("log_steps"),
                                    batch_size=reader.batch_size,
                                    job_type="auc"))
    # summary prediction hook 
    if ctx.get_config("summary"):
      ctx.add_hook(utils.summary.PredictionHook(auc, global_step, ctx.get_config(), auc_vars, reader.batch_size))
    # save auc hook
    if ctx.get_config("auc"):
      ctx.add_hook(utils.save_auc.SaveAucHook(auc, global_step,
                                              run_interval=ctx.get_config("log_steps")))
    #  auc tracer
    if ctx.get_config("tracer.bin"):
      ctx.add_hook(utils.tracer_bin.TracerHook(ctx.get_config("tracer.bin"), datas))
    # csv tracer
    if ctx.get_config("trace"):
      ctx.add_hook(utils.tracer_csv.TracerHook(ctx.get_config(), global_step))
    # profiler
    if ctx.get_config("profiler"):
      ctx.add_hook(utils.profiler.ProfilerHook(ctx.get_config()))
    # create session and run
    with ctx.session() as sess:
      if ctx.get_task_index() == 0:
        if zero_ops:
          sess.run(zero_ops)
          if len(zero_ops) != 4:
            print("Warn: clear " + str(len(zero_ops)) + " auc variables.")
      else:
        import time
        time.sleep(60)
      while not sess.should_stop():
        if feed_dict is not None:
          feed = {k: (v() if callable(v) else v) for k, v in feed_dict.items()}
        else:
          feed = None
        sess.run(auc, feed_dict=feed)

########### model convert
def model_convert_default(inference_fn=None):
  
  import tf_model_convert
  if ctx.get_config("ps_type") == "ps_plus":
    ctx.get_context()
    tf_model_convert.model_convert_plus(ctx.get_task_index(), 
                                          ctx.get_config("worker", "instance_num"), 
                                          ctx.get_config("checkpoint"), 
                                          ctx.get_config("model_convert"))
  elif ctx.get_config("ps_type") == "ps_native":
    datas, reader = read_data()
    with ctx.graph():
      global_step = tf.contrib.framework.get_or_create_global_step()
      # inference
      loss, auc = inference_fn(datas)
      # create session and run
      with ctx.session() as sess:
        tf_model_convert.model_convert_native(ctx.get_task_index(), 
                                              ctx.get_config("worker", "instance_num"), 
                                              ctx.get_config("checkpoint"), 
                                              ctx.get_config("model_convert"), 
                                              sess)