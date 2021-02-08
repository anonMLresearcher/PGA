import numpy as np
import tensorflow as tf
import tf_sort


def swd_proj_src(x, Theta, nation_idx_where):
    
 
    x = tf.matmul(x,Theta, transpose_b=True)
    x = tf.concat([x,tf.zeros((2,tf.shape(x)[1]))],0)
    x_list=[]
    idx_list = []
 
    for e_idx in nation_idx_where:
        e_x_i = tf.gather(x,e_idx,axis=0)
        e_x = tf.contrib.framework.sort(e_x_i,axis=0)
        idx_sort = tf_sort.argsort(e_x_i,axis=0)
        idx_sort = tf.stop_gradient(idx_sort)
    
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

def swd_proj_pos_neg(pos,neg, Theta, nation_idx_where,neg_num):
    
 
    pos = tf.matmul(pos,Theta, transpose_b=True)
    neg = tf.matmul(neg,Theta, transpose_b=True)
    neg = tf.reshape(neg, [-1, neg_num, tf.shape(neg)[1]])
    x = [pos]
    for i in range(neg_num):
        x.append(neg[:,i,:])
        
    xx = []
    for e in x:
        e = tf.concat([e,tf.zeros((2,tf.shape(e)[1]))],0)
        xx.append(e)
    
     
    x_list=[]
 
    idx_list = []
 
    for e_idx in nation_idx_where:
        data_list = []
        for x in xx:
            e_x = tf.gather(x,e_idx,axis=0)
            data_list.append(e_x)
        e_x_i = tf.concat(data_list,0)
            
        e_x = tf.contrib.framework.sort(e_x_i,axis=0)
   
        idx_sort = tf_sort.argsort(e_x_i,axis=0)
        idx_sort = tf.stop_gradient(idx_sort)
 
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