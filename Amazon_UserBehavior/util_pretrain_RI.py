import numpy as np
import tensorflow as tf
import tf_sort


def get_initializer(init_val=1, dtype=tf.float32, stddev=0.1, value=0.0):
    if init_val == 0:
        return tf.constant_initializer(value=value, dtype=dtype)
    elif init_val == 1:
        return tf.truncated_normal_initializer(dtype=dtype, stddev=stddev)
    elif init_val == 2:
        # factor*[-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]
        # stddev=factor/sqrt(N)
        # where factor=input stddev!!!!!!!!!!!!!!!
        return tf.uniform_unit_scaling_initializer(factor=stddev, seed=10, dtype=tf.float32)
    else:
        return None

def full_connect(train_inputs, weights_shape, biases_shape, no_biases, ps_num, init_val, stddev):
    # weights
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev),
                                  regularizer=tf.nn.l2_loss,
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  regularizer=tf.nn.l2_loss,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev))

    if no_biases:
        # matmul
        train = tf.matmul(train_inputs, weights)
        return train
    else:
        # biases
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=get_initializer(init_val=0, value=0.0002))
        # matmul
        train = tf.matmul(train_inputs, weights) + biases
        return train

def mapping_net(id_deep, ps_num, init_val, stddev, sp_name):
    with tf.variable_scope("mapping_layer1_%s"%sp_name):
        x = full_connect(id_deep, [32, 32], [32], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        x=tf.nn.l2_normalize(x,axis=1)
        
#     with tf.variable_scope("mapping_layer2_%s"%sp_name):
#         x = full_connect(x, [32, 32], [32], False, ps_num, init_val, stddev)
#         x = tf.nn.elu(x)
         
    return x



def swd_proj_src(xx, Theta, nation_idx_where):
     
    
    x_list=[]
    idx_list = []
 
    for i_idx,e_idx in enumerate(nation_idx_where):
        x = tf.matmul(xx[i_idx],Theta, transpose_b=True)
        x = tf.concat([x,tf.zeros((2,tf.shape(x)[1]))],0)
    
        e_x_i = tf.gather(x,e_idx,axis=0)
        e_x = tf.contrib.framework.sort(e_x_i,axis=0)
        idx_sort = tf_sort.argsort(e_x_i,axis=0)
    
        x_list.append(e_x)
        idx_list.append(idx_sort)
   
    return x_list,idx_list 

def swd_proj_src_with_idx(x,idx_list, nation_idx_where):
 
    x = tf.concat([x,tf.zeros((2,tf.shape(x)[1]))],0)
    x_list=[]
 
    for i_idx,e_idx in enumerate(nation_idx_where):
        e_x_i = tf.gather(x,e_idx,axis=0)
        idx_sort = tf.reshape(idx_list[i_idx],[-1])
        e_x = tf.reshape(tf.gather(e_x_i,idx_sort),[tf.shape(e_x_i)[0],-1,tf.shape(e_x_i)[1]])
        x_list.append(e_x)
 
    return x_list 


def swd_proj_pos_neg(pos_list,neg_list, Theta, nation_idx_where,neg_num):
    
 
#     pos = tf.matmul(pos,Theta, transpose_b=True)
#     neg = tf.matmul(neg,Theta, transpose_b=True)
    
    xxx = []
    
    for pos,neg in zip(pos_list,neg_list):
    
        neg = tf.reshape(neg, [-1, neg_num, tf.shape(neg)[1]])
        x = [pos]
        for i in range(neg_num):
            x.append(neg[:,i,:])

        xx = []
        for e in x:
            e = tf.concat([e,tf.zeros((2,tf.shape(e)[1]))],0)
            xx.append(e)
            
        xxx.append(xx)
    
     
    x_list=[]
 
    idx_list = []
 
    for i_idx,e_idx in enumerate(nation_idx_where):
        data_list = []
        for x in xxx[i_idx]:
            x = tf.matmul(x,Theta, transpose_b=True)
            e_x = tf.gather(x,e_idx,axis=0)
            data_list.append(e_x)
        e_x_i = tf.concat(data_list,0)
            
        e_x = tf.contrib.framework.sort(e_x_i,axis=0)
   
        idx_sort = tf_sort.argsort(e_x_i,axis=0)
 
        x_list.append(e_x)
 
        idx_list.append(idx_sort)
     
        
    # loss = tf.reduce_mean(tf.abs(y_pred_proj-y_true_proj))
    
    return x_list ,idx_list 
 

def swd_proj_pos_neg_with_idx(pos,neg, idx_list, nation_idx_where,neg_num):
    
  
    neg = tf.reshape(neg, [-1, neg_num, tf.shape(neg)[1]])
    x = [pos]
    for i in range(neg_num):
        x.append(neg[:,i,:])
        
    xx = []
    for e in x:
        e = tf.concat([e,tf.zeros((2,tf.shape(e)[1]))],0)
        xx.append(e)
    
     
    x_list=[]
  
    for i_idx,e_idx in enumerate(nation_idx_where):
        data_list = []
        for x in xx:
            e_x = tf.gather(x,e_idx,axis=0)
            data_list.append(e_x)
        e_x_i = tf.concat(data_list,0)
            
        idx_sort = tf.reshape(idx_list[i_idx],[-1])
        e_x = tf.reshape(tf.gather(e_x_i,idx_sort),[tf.shape(e_x_i)[0],-1,tf.shape(e_x_i)[1]])
        x_list.append(e_x)
  
    
    return x_list  

def mapping_fun(current_id_ays, ps_num, dense_ini_val, dense_stddev,nation_list):
    current_id_ays_nation_list = []
    for e_nation in nation_list:
        current_id_ays_nation=mapping_net(current_id_ays, ps_num, dense_ini_val, dense_stddev, e_nation)
        current_id_ays_nation_list.append(current_id_ays_nation)
    return current_id_ays_nation_list