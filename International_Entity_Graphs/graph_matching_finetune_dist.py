import numpy as np
import tensorflow as tf
import tf_sort

def gm_dist(a,b):
    return tf.abs(a-b)

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


def full_connect(train_inputs, weights_shape, biases_shape, no_biases, ps_num, init_val, stddev,trainable=True):
    # weights
    if ps_num > 1:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev),
                                  regularizer=tf.nn.l2_loss,
                                  partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num),
                                  trainable=trainable)
    else:
        weights = tf.get_variable("weights",
                                  weights_shape,
                                  regularizer=tf.nn.l2_loss,
                                  initializer=get_initializer(init_val=init_val, stddev=stddev),
                                  trainable=trainable)

    if no_biases:
        # matmul
        train = tf.matmul(train_inputs, weights)
        return train
    else:
        # biases
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=get_initializer(init_val=0, value=0.0002),
                                 trainable=trainable)
        # matmul
        train = tf.matmul(train_inputs, weights) + biases
        return train

def gate_gwd_net(id_deep, ps_num, init_val, stddev, sp_name):
    with tf.variable_scope("gwd_layer1_%s"%sp_name):
        x = full_connect(tf.stop_gradient(id_deep), [96, 32], [32], False, ps_num, init_val, stddev,trainable=False)
        x = tf.nn.elu(x)
        
    with tf.variable_scope("gwd_layer2_%s"%sp_name):
        x = full_connect(x, [32, 6], [6], False, ps_num, init_val, stddev,trainable=False)
        x = tf.sigmoid(x)
         
    return x

def greater_than_zero(x):
    x=tf.maximum(0.0*x,x)
    x = tf.sign(x)
    return x


def gate_loss(g,th=0.8,flag_above=True):
    entropy = tf.reduce_mean(-g*tf.log(g+1e-6)-(1.-g)*tf.log(1.-g+1e-6))
    
    g = tf.reduce_mean(g,0)
    if flag_above:
        flag = tf.stop_gradient(greater_than_zero(g-th))
        loss = -tf.log(g+1e-6) * (1.-flag) 
    else:
        loss =  g*tf.log(g+1e-6)+(1.-g)*tf.log(1.-g+1e-6)
    
    return entropy,tf.reduce_mean(loss)

def list_reverse(x,rank):
    y =[]
    for e in x:
        # rank = tf.rank(e)
        if rank==1:
            y.append(e[-1::-1])
        elif rank==2:
            y.append(e[-1::-1,:])
        elif rank==3:
            y.append(e[-1::-1,:,:])
    return y

def tf_var(x,axis,keepdims=False):
    m = tf.reduce_mean(x,axis,keepdims=True)
    return tf.reduce_mean(tf.square(x-m),axis,keepdims=keepdims)

def swd_loss(x,y,nation_exist,idx,idx_norm,x_g,y_g,nation_weights=None,
gm_loss_weight=1,flag_dist=1.,var_weight=1e-6,gate_weight=1.,
gate_th=0.8,flag_gate_above=True,same_nation_weight=1.,
flag_uniform_nation_weight=False,contain_same_cate=None):

    if contain_same_cate is None:
        nation_exist = tf.reshape(nation_exist,[-1])
        n_nation = tf.shape(nation_exist)[0]
        contain_same_cate = tf.ones((n_nation,n_nation))

    if flag_uniform_nation_weight or nation_weights is None:
        nation_weights = tf.ones_like(nation_exist)
        nation_weights /= tf.reduce_sum(nation_weights)

    
    
    loss_num=0.0

    dist_list =  []
    entropy_list = []
    th_loss_list = []
    gate_prob_list = []
    loss_var_list = []
    for i_level in range(len(x)):
        ex,ey=x[i_level],y[i_level]
        ex_g,ey_g=x_g[i_level],y_g[i_level]
       
        e_x,_ = ex
        e_y,_ = ey
        e_xr = list_reverse(e_x,2)
        e_yr = list_reverse(e_y,2)
        _,e_x_g  = ex_g
        _,e_y_g  = ey_g
        e_xr_g = list_reverse(e_x_g,3)
        e_yr_g = list_reverse(e_y_g,3)
        
        dist_sum = 0.0
        loss_gate = 0.0
        loss_var_sum=0.0
        cont = 0.0
        
        e_x_list = []
        e_x_g_list = []
        e_y_list = []
        e_y_g_list = []
        e_xr_list = []
        e_xr_g_list = []
        e_yr_list = []
        e_yr_g_list = []
        
        for i_nation in range(len(e_x)):
            idx_i = tf.cast(idx_norm*tf.to_float(tf.shape(e_x[i_nation])[0]),tf.int32)
     
            e_x_i = tf.gather(e_x[i_nation],idx_i,axis=0)
            e_x_i_g = tf.gather(e_x_g[i_nation],idx_i,axis=0)
            e_xr_j = tf.gather(e_xr[i_nation],idx_i,axis=0)
            e_xr_j_g = tf.gather(e_xr_g[i_nation],idx_i,axis=0)
            e_x_list.append(e_x_i)
            e_x_g_list.append(e_x_i_g)
            e_xr_list.append(e_xr_j)
            e_xr_g_list.append(e_xr_j_g)
            
        for j_nation in range(len(e_y)):
            idx_j = tf.cast(idx_norm*tf.to_float(tf.shape(e_y[j_nation])[0]),tf.int32)
        
            e_y_j = tf.gather(e_y[j_nation],idx_j,axis=0)
            e_y_j_g = tf.gather(e_y_g[j_nation],idx_j,axis=0)
            e_yr_j = tf.gather(e_yr[j_nation],idx_j,axis=0)
            e_yr_j_g = tf.gather(e_yr_g[j_nation],idx_j,axis=0)
            e_y_list.append(e_y_j)
            e_y_g_list.append(e_y_j_g)
            e_yr_list.append(e_yr_j)
            e_yr_g_list.append(e_yr_j_g)
        
        for i_nation in range(len(e_x)):
            e_x_i = e_x_list[i_nation]
            e_x_i_g = e_x_g_list[i_nation] 
            nation_exist_i = nation_exist[i_nation]
            cont = cont + 1.
            for j_nation in range(len(e_x)):
                e_y_j = tf.stop_gradient(e_y_list[j_nation])
                e_y_j_g = tf.stop_gradient(e_y_g_list[j_nation])
                e_yr_j = tf.stop_gradient(e_yr_list[j_nation])
                e_yr_j_g = tf.stop_gradient(e_yr_g_list[j_nation])

                dist_pos = tf.reduce_mean(gm_dist(e_x_i,e_y_j)*tf.stop_gradient(e_x_i_g[:,:,j_nation]*e_y_j_g[:,:,i_nation]))
                dist_neg = tf.reduce_mean(gm_dist(e_x_i,e_yr_j)*tf.stop_gradient(e_x_i_g[:,:,j_nation]*e_yr_j_g[:,:,i_nation]))
                dist = tf.minimum(dist_pos,dist_neg) * nation_exist_i * nation_exist[j_nation]
                entropy_i,th_loss_i = gate_loss(e_x_i_g[:,:,j_nation],th=gate_th,flag_above=flag_gate_above)
                entropy_j,th_loss_j = gate_loss(e_y_j_g[:,:,i_nation],th=gate_th,flag_above=flag_gate_above)
                var_i = tf_var(e_x_i,0)
                loss_var = tf.reduce_mean( -tf.log(var_i+1e-6))
                
                dist = dist*gm_loss_weight
                loss_var = loss_var*gm_loss_weight

                dist_list.append(dist* nation_weights[j_nation])
                entropy_list.append((entropy_i*0.5+entropy_j*0.5)* nation_weights[j_nation])
                th_loss_list.append((th_loss_i*0.5+th_loss_j*0.5)* nation_weights[j_nation])
                gate_prob_list.append((tf.reduce_mean(e_x_i_g[:,:,j_nation])*0.5+tf.reduce_mean(e_y_j_g[:,:,i_nation])*0.5) )
                loss_var_list.append((loss_var)* nation_weights[j_nation])
                
                if i_nation==j_nation:
                    pair_nation_weight = same_nation_weight * contain_same_cate[i_nation,j_nation]
                else:
                    pair_nation_weight = 1. * contain_same_cate[i_nation,j_nation]

                dist_sum = dist_sum + 0.5 * dist * nation_weights[j_nation] * pair_nation_weight
                loss_gate = loss_gate + 0.5*(entropy_i + th_loss_i + entropy_j + th_loss_j) * nation_weights[j_nation] * pair_nation_weight
                loss_var_sum = loss_var_sum + 0.5 * loss_var * nation_weights[j_nation] * pair_nation_weight

        for i_nation in range(len(e_x)):
            e_x_i = e_y_list[i_nation]
            e_x_i_g = e_y_g_list[i_nation] 
            nation_exist_i = nation_exist[i_nation]
            cont = cont + 1.
            for j_nation in range(len(e_x)):
                e_y_j = tf.stop_gradient(e_x_list[j_nation])
                e_y_j_g = tf.stop_gradient(e_x_g_list[j_nation])
                e_yr_j = tf.stop_gradient(e_xr_list[j_nation])
                e_yr_j_g = tf.stop_gradient(e_xr_g_list[j_nation])

                dist_pos = tf.reduce_mean(gm_dist(e_x_i,e_y_j)*tf.stop_gradient(e_x_i_g[:,:,j_nation]*e_y_j_g[:,:,i_nation]))
                dist_neg = tf.reduce_mean(gm_dist(e_x_i,e_yr_j)*tf.stop_gradient(e_x_i_g[:,:,j_nation]*e_yr_j_g[:,:,i_nation]))
                dist = tf.minimum(dist_pos,dist_neg) * nation_exist_i * nation_exist[j_nation]
                entropy_i,th_loss_i = gate_loss(e_x_i_g[:,:,j_nation],th=gate_th,flag_above=flag_gate_above)
                entropy_j,th_loss_j = gate_loss(e_y_j_g[:,:,i_nation],th=gate_th,flag_above=flag_gate_above)
                var_i = tf_var(e_x_i,0)
                loss_var = tf.reduce_mean( -tf.log(var_i+1e-6))
                
                dist = dist*gm_loss_weight
                loss_var = loss_var*gm_loss_weight

                dist_list.append(dist* nation_weights[j_nation])
                entropy_list.append((entropy_i*0.5+entropy_j*0.5)* nation_weights[j_nation])
                th_loss_list.append((th_loss_i*0.5+th_loss_j*0.5)* nation_weights[j_nation])
                gate_prob_list.append((tf.reduce_mean(e_x_i_g[:,:,j_nation])*0.5+tf.reduce_mean(e_y_j_g[:,:,i_nation])*0.5) )
                loss_var_list.append((loss_var)* nation_weights[j_nation])

                if i_nation==j_nation:
                    pair_nation_weight = same_nation_weight * contain_same_cate[i_nation,j_nation]
                else:
                    pair_nation_weight = 1. * contain_same_cate[i_nation,j_nation]
                
                dist_sum = dist_sum + 0.5 * dist * nation_weights[j_nation] * pair_nation_weight
                loss_gate = loss_gate + 0.5*(entropy_i + th_loss_i + entropy_j + th_loss_j) * nation_weights[j_nation] * pair_nation_weight
                loss_var_sum = loss_var_sum + 0.5 * loss_var * nation_weights[j_nation] * pair_nation_weight
                
        loss_num = loss_num + dist_sum/cont   + var_weight * flag_dist* loss_var_sum/cont

        break

    
    dist = tf.reduce_mean(tf.stack(dist_list))
    entropy = tf.reduce_mean(tf.stack(entropy_list))
    th_loss = tf.reduce_mean(tf.stack(th_loss_list))
    gate_prob =  tf.reduce_mean(tf.stack(gate_prob_list))
    loss_var =  tf.reduce_mean(tf.stack(loss_var_list))
    return loss_num,[dist,entropy,th_loss,gate_prob,loss_var]

 
def sgwd_loss(x,y,nation_exist,gwd_idx,gwd_idx_norm,x_g,y_g,ps_num, init_val, stddev,mode='gwd',
nation_weights=None,gm_loss_weight=1,flag_dist=1.,
var_weight=1e-6,gate_weight=1.,gate_th=0.8,flag_gate_above=True,
same_nation_weight=1.,flag_uniform_nation_weight=False,contain_same_cate=None):

    if contain_same_cate is None:
        nation_exist = tf.reshape(nation_exist,[-1])
        n_nation = tf.shape(nation_exist)[0]
        contain_same_cate = tf.ones((n_nation,n_nation))

    if flag_uniform_nation_weight or nation_weights is None:
        nation_weights = tf.ones_like(nation_exist)
        nation_weights /= tf.reduce_sum(nation_weights)

    loss_num=0.0
    gwd_idx_norm_left,gwd_idx_norm_right=gwd_idx_norm 
    
    dist_list =  []
    entropy_list = []
    th_loss_list = []
    gate_prob_list = []
    loss_var_list = []
    
    for i_level in range(len(x)):
   
        ex,ey=x[i_level],y[i_level]
        ex_g,ey_g=x_g[i_level],y_g[i_level]
       
        e_x,_ = ex
        e_y,_ = ey
        e_xr = list_reverse(e_x,2)
        e_yr = list_reverse(e_y,2)
        e_x_g,_  = ex_g
        e_y_g,_  = ey_g
        e_xr_g = list_reverse(e_x_g,3)
        e_yr_g = list_reverse(e_y_g,3)
        
        dist_sum = 0.0
        loss_gate = 0.0
        loss_var_sum=0.0
        cont = 0.0
        
        e_x_list = []
        e_x_g_list = []
        e_xr_list = []
        e_xr_g_list = []
        
        e_y_list = []
        e_y_g_list = []
        e_yr_list = []
        e_yr_g_list = []
        
        for i_nation in range(len(e_x)):
            idx_i = tf.cast(gwd_idx_norm_left*tf.to_float(tf.shape(e_x[i_nation])[0]),tf.int32)
     
            e_x_i = tf.gather(e_x[i_nation],idx_i,axis=0)
            e_x_i_g = tf.gather(e_x_g[i_nation],idx_i,axis=0)
            e_xr_i = tf.gather(e_xr[i_nation],idx_i,axis=0)
            e_xr_i_g = tf.gather(e_xr_g[i_nation],idx_i,axis=0)
             
            e_x_list.append(e_x_i)
            e_x_g_list.append(e_x_i_g)
            e_xr_list.append(e_xr_i)
            e_xr_g_list.append(e_xr_i_g)
          
        for j_nation in range(len(e_y)):
            idx_j = tf.cast(gwd_idx_norm_right*tf.to_float(tf.shape(e_y[j_nation])[0]),tf.int32)
        
            e_y_j = tf.gather(e_y[j_nation],idx_j,axis=0)
            e_y_j_g = tf.gather(e_y_g[j_nation],idx_j,axis=0)
            e_yr_j = tf.gather(e_yr[j_nation],idx_j,axis=0)
            e_yr_j_g = tf.gather(e_yr_g[j_nation],idx_j,axis=0)
            
            e_y_list.append(e_y_j)
            e_y_g_list.append(e_y_j_g)
            e_yr_list.append(e_yr_j)
            e_yr_g_list.append(e_yr_j_g)
            
        gate_list = []
        for i_nation in range(len(e_x)):
            e_x_i_g = e_x_g_list[i_nation]
            e_y_i_g = e_y_g_list[i_nation]
            
            e_xr_i_g = e_xr_g_list[i_nation]
            e_yr_i_g = e_yr_g_list[i_nation]
            
            e_x_i_g_flatten = tf.reshape(e_x_i_g,[-1,tf.shape(e_x_i_g)[2]])
            e_y_i_g_flatten = tf.reshape(e_y_i_g,[-1,tf.shape(e_y_i_g)[2]])
            
            e_xr_i_g_flatten = tf.reshape(e_xr_i_g,[-1,tf.shape(e_xr_i_g)[2]])
            e_yr_i_g_flatten = tf.reshape(e_yr_i_g,[-1,tf.shape(e_yr_i_g)[2]])
            
            feature_i = tf.concat([e_x_i_g_flatten,e_y_i_g_flatten,e_x_i_g_flatten*e_y_i_g_flatten],1)
            feature_r_i = tf.concat([e_xr_i_g_flatten,e_yr_i_g_flatten,e_xr_i_g_flatten*e_yr_i_g_flatten],1)
            
            if i_level == 0:
                sp_name='bottom'
            else:
                sp_name='upper'

            gate_i = gate_gwd_net(feature_i, ps_num, init_val, stddev,sp_name)
            gate_r_i = gate_gwd_net(feature_r_i, ps_num, init_val, stddev,sp_name)
            # else:
            #     gate_i = gate_cgwd_net(feature_i, ps_num, init_val, stddev)
            #     gate_r_i = gate_cgwd_net(feature_r_i, ps_num, init_val, stddev)
                
            gate_i = tf.reshape(gate_i,[tf.shape(e_x_i_g)[0],-1,tf.shape(gate_i)[1]])
            gate_r_i = tf.reshape(gate_r_i,[tf.shape(e_xr_i_g)[0],-1,tf.shape(gate_r_i)[1]])
                 
            gate_list.append([gate_i,gate_r_i])
        
        for i_nation in range(len(e_x)):
            e_x_i = e_x_list[i_nation]
            e_y_i = e_y_list[i_nation]
             
            gate_i,_ = gate_list[i_nation]
            dist_i = gm_dist(e_x_i,e_y_i)
            
            nation_exist_i = nation_exist[i_nation]
            cont = cont + 1.
            for j_nation in range(len(e_x)):
                 
                e_x_j  = e_x_list[j_nation]
                e_y_j  = e_y_list[j_nation]
                
                e_xr_j  = e_xr_list[j_nation]
                e_yr_j  = e_yr_list[j_nation]
                 
                gate_j,gate_r_j = gate_list[j_nation]
                gate_j = tf.stop_gradient(gate_j )
                gate_r_j = tf.stop_gradient( gate_r_j)
                
                dist_j = tf.stop_gradient( gm_dist(e_x_j,e_y_j) )
                dist_r_j = tf.stop_gradient( gm_dist(e_xr_j,e_yr_j) )
                 
                dist_pos = tf.reduce_mean(gm_dist(dist_i,dist_j)*tf.stop_gradient(gate_i[:,:,j_nation]*gate_j[:,:,i_nation]))
                dist_neg = tf.reduce_mean(gm_dist(dist_i,dist_r_j)*tf.stop_gradient(gate_i[:,:,j_nation]*gate_r_j[:,:,i_nation]))
                
                flag = tf.stop_gradient(greater_than_zero(dist_pos-dist_neg))
                
                entropy_ij,th_loss_ij = gate_loss(gate_i[:,:,j_nation],th=gate_th,flag_above=flag_gate_above) 
                entropy_ji,th_loss_ji =gate_loss(gate_j[:,:,i_nation],th=gate_th,flag_above=flag_gate_above)
                entropy_rji,th_loss_rji = gate_loss(gate_r_j[:,:,i_nation],th=gate_th,flag_above=flag_gate_above)

                gate_prob_ij = tf.reduce_mean(gate_i[:,:,j_nation])
                gate_prob_ji = tf.reduce_mean(gate_j[:,:,i_nation])
                gate_prob_rji = tf.reduce_mean(gate_r_j[:,:,i_nation])
                
                loss_gate_pos =  0.5 * (entropy_ij+th_loss_ij) + 0.5 * (entropy_ji+th_loss_ji)
                loss_gate_neg =  0.5 * (entropy_ij+th_loss_ij) + 0.5 * (entropy_rji+th_loss_rji)

                var_x = tf_var(e_x_i,0)
                var_y = tf_var(e_y_i,0)
                loss_var = tf.reduce_mean( -0.5 * tf.log(var_x+1e-6)- 0.5 * tf.log(var_y+1e-6))
                
                 
                loss_gate = loss_gate +  ( (1.-flag) * loss_gate_pos +  flag * loss_gate_neg ) * nation_weights[j_nation]
                
                dist = tf.minimum(dist_pos,dist_neg) * nation_exist_i * nation_exist[j_nation]
                
                dist = dist*gm_loss_weight
                loss_var = loss_var*gm_loss_weight

                dist_list.append((dist)* nation_weights[j_nation])
                entropy_list.append(((1.-flag) *(entropy_ij*0.5+entropy_ji*0.5) + flag *(entropy_ij*0.5+entropy_rji*0.5))* nation_weights[j_nation])
                th_loss_list.append(((1.-flag) *(th_loss_ij*0.5+th_loss_ji*0.5) + flag *(th_loss_ij*0.5+th_loss_rji*0.5))* nation_weights[j_nation])
                gate_prob_list.append(((1.-flag) *(gate_prob_ij*0.5+gate_prob_ji*0.5) + flag *(gate_prob_ij*0.5+gate_prob_rji*0.5)) )
                loss_var_list.append((loss_var)* nation_weights[j_nation])
                
                if i_nation==j_nation:
                    pair_nation_weight = same_nation_weight * contain_same_cate[i_nation,j_nation]
                else:
                    pair_nation_weight = 1. * contain_same_cate[i_nation,j_nation]

                dist_sum = dist_sum + 0.5*  dist * nation_weights[j_nation] * pair_nation_weight
                loss_var_sum = loss_var_sum + 0.5 * loss_var * nation_weights[j_nation] * pair_nation_weight

                
        loss_num = loss_num + 2.* ( dist_sum/cont   + var_weight * flag_dist * loss_var_sum )
         
        break
    
    dist = tf.reduce_mean(tf.stack(dist_list))
    entropy = tf.reduce_mean(tf.stack(entropy_list))
    th_loss = tf.reduce_mean(tf.stack(th_loss_list))
    gate_prob =  tf.reduce_mean(tf.stack(gate_prob_list))
    loss_var =  tf.reduce_mean(tf.stack(loss_var_list))
    return loss_num,[dist,entropy,th_loss,gate_prob,loss_var]