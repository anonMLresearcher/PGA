from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tf_euler

import tensorflow as tf
from tensorflow.python.util import nest
import tf_context as ctx
import graph_tag
import base_runner
import numpy as np

def get_high64bit(tensor):
    # 1-D tensor
    # input: [1, 2, 3, 4, 5, 6], return: [1, 3, 5]
    tensor = tf.reshape(tensor, [-1, 2])
    return tensor[:, 0]

def tohash128(sp_tensors):
    for i in range(len(sp_tensors)):
        sp = sp_tensors[i]

        indices = sp.indices
        n = tf.shape(indices)[0]
        dim = tf.shape(indices)[1]
        indices = tf.multiply(indices, [1, 2])
        zero_indices = tf.add(indices, [0, 1])
        new_indices = tf.stack([indices, zero_indices], 0)
        new_indices = tf.transpose(new_indices, perm=[1, 0, 2])
        new_indices = tf.reshape(new_indices, [2 * n, dim])

        values = sp.values
        zero = tf.zeros_like(values)
        new_values = tf.stack([values, zero])
        new_values = tf.transpose(new_values)
        new_values = tf.reshape(new_values, [-1])

        shape = sp.dense_shape
        new_shape = tf.multiply(shape, [1, 2])

        sp_tensors[i] = tf.SparseTensor(indices=new_indices, values=new_values, dense_shape=new_shape)

def sparse_embedding(sp_tensors, input_dimensions, embedding_dimension, ps_num, init_val, stddev, names):
    l = []
    for i in range(len(sp_tensors)):
        with tf.variable_scope(names[i]):
            print("sparse_embedding name:" + names[i] + " dim:" + str(input_dimensions[i]))
            rst = full_connect_sparse(sp_tensors[i], [input_dimensions[i], embedding_dimension[i]], None, True, ps_num,
                                      None, init_val, stddev)
            l.append(rst)
    return l

def dense_embedding(sp_tensors, input_dimensions, embedding_dimension, ps_num, init_val, stddev, names):
    l = []
    print('len(dense_tensors)',len(sp_tensors))
    print('len(names)',len(names))
    for i in range(len(sp_tensors)):
        with tf.variable_scope(names[i]):
            print("dense_embedding name:" + names[i] + " dim:" + str(input_dimensions[i]))
            rst = full_connect(sp_tensors[i], [input_dimensions[i], embedding_dimension[i]], None, True, ps_num, init_val, stddev)
            rst = tf.nn.elu(rst)
            l.append(rst)
    return l

def get_neighbors_inf(src,part_q_feature_names,part_i_feature_names, i_nei_cnt=5, q_nei_cnt=5):
    # q_part_features = [8]
    # i_part_features = [1]

    src_i_nodes, _, _ = tf_euler.sample_neighbor(src, edge_types=['1', '3'], count=i_nei_cnt)
    src_i_nodes_filled = tf_euler.get_sparse_feature(src_i_nodes, part_i_feature_names)
    src_q_nodes, _, _ = tf_euler.sample_neighbor(src, edge_types=['2'], count=q_nei_cnt)
    src_q_nodes_filled = tf_euler.get_sparse_feature(src_q_nodes, part_q_feature_names)

    tohash128(src_i_nodes_filled)
    tohash128(src_q_nodes_filled)
    return src_i_nodes_filled, src_q_nodes_filled

def sample_pos_n_feature_inf(src,full_features,q_part_features,i_part_features,c_part_features,full_features_dense, full_feature_dims_dense):
    # full_features = [8, 9, 6, 1, 2, 3, 4, 5, 7]
    
    self_f = get_feature(src,full_features)
    src_dense = tf_euler.get_dense_feature(src, full_features_dense, full_feature_dims_dense)

    q_f,i_f,c_f = get_neighbor(src,q_part_features,i_part_features,c_part_features)
    feature_dic = {'self':self_f,'q':q_f,'i':i_f,'c':c_f}
    return feature_dic,src_dense
  
def get_feature(src,full_features):
    src_filled = tf_euler.get_sparse_feature(src, full_features)
    tohash128(src_filled)
    return src_filled

def get_neighbor(x,q_part_features,i_part_features,c_part_features, q_nei_cnt=5,i_nei_cnt=5,c_nei_cnt=5 ):
 

    q_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['2','5'], count=i_nei_cnt)
    i_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=[  '3','7'], count=i_nei_cnt)
    c_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['46','66' ], count=i_nei_cnt)
    
     
    q_f = get_feature(q_nodes,q_part_features)
    i_f = get_feature(i_nodes,i_part_features)
    c_f = get_feature(c_nodes,c_part_features)
    
    return q_f,i_f,c_f
      

def fake_convolution(node, q_node_nei,i_node_nei,c_node_nei, q_num=5, i_num=5,c_num=5,   q_dim=8,i_dim=8,c_dim=8):
    q_node_nei_s = tf.reshape(q_node_nei, [-1, q_num, q_dim])
    q_nei = tf.reduce_mean(q_node_nei_s, 1)
    i_node_nei_s = tf.reshape(i_node_nei, [-1, i_num, i_dim])
    i_nei = tf.reduce_mean(i_node_nei_s, 1)
    c_node_nei_s = tf.reshape(c_node_nei, [-1, c_num, c_dim])
    c_nei = tf.reduce_mean(c_node_nei_s, 1)
     
    res = tf.concat([node, q_nei, i_nei, c_nei], 1)
    return res


def inference(tensors, keep_prob=1.0, ps_num=1):
    id_embedding_dim = 24

    embedding_ini_val = 1
    embedding_stddev = 0.0002

    dense_ini_val = 2
    dense_stddev = 0.36
 

    nation_list=['ES','FR','NL','RU','US']
    feature_list=[
        {
            'feature_name':'venture_category_name_en',
            'feature_dim':int(252*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'category',
            'feature_dim':int(252*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'query',
            'feature_dim':int(13259196*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_id',
            'feature_dim':int(232766476*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature19',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature34',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature35',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature36',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature40',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature41',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature42',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature2',
            'feature_dim':int(11*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature3',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature4',
            'feature_dim':int(6*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature5',
            'feature_dim':int(3*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature6',
            'feature_dim':int(33*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature7',
            'feature_dim':int(7*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature8',
            'feature_dim':int(50*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature32',
            'feature_dim':int(8*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'user_feature33',
            'feature_dim':int(8*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'nation',
            'feature_dim':6,
            'emb_dim':8
        }
    ]
    
    
    dic = {
        '1':	7,
        '2':	50
    }
    for e_level in ['1','2']:
        feature_list.append(
            {
                'feature_name':'venture_category%s_name_en' % e_level,
                'feature_dim':int(dic[e_level]*1.1),
                'emb_dim':8
            }
        )

      
    full_feature_dims = []
    full_feature_names = []
    full_emb_dim = []

    for e in feature_list:
        full_feature_names.append(e['feature_name'])
        full_feature_dims.append(e['feature_dim'])
        full_emb_dim.append(e['emb_dim'])

    full_feature_names = (full_feature_names)
 
 
    part_q_feature_names = ['query' ]
    part_q_feature_dims = [ int(13259196*1.1) ]
    part_q_emb_dim = [8 ]

    part_i_feature_names = ['item_id' ]
    part_i_feature_dims = [int(232766476*1.1) ]
    part_i_emb_dim = [8 ]
    
    part_c_feature_names = ['venture_category_name_en' ]
    part_c_feature_dims = [int(252*1.1) ]
    part_c_emb_dim = [8 ]
     
    full_emb_dim_sum=np.sum(full_emb_dim)+np.sum(part_q_emb_dim)+np.sum(part_i_emb_dim)+np.sum(part_c_emb_dim)
    part_q_emb_dim_sum=np.sum(part_q_emb_dim)
    part_i_emb_dim_sum=np.sum(part_i_emb_dim)
    part_c_emb_dim_sum=np.sum(part_c_emb_dim)
    
    dic_info={
        'self':
        {
            'feature_names':full_feature_names,
            'feature_dims':full_feature_dims,
            'emb_dim':full_emb_dim
        },
        'q':
        {
            'feature_names':part_q_feature_names,
            'feature_dims':part_q_feature_dims,
            'emb_dim':part_q_emb_dim
        },
        'i':
        {
            'feature_names':part_i_feature_names,
            'feature_dims':part_i_feature_dims,
            'emb_dim':part_i_emb_dim
        },
        'c':
        {
            'feature_names':part_c_feature_names,
            'feature_dims':part_c_feature_dims,
            'emb_dim':part_c_emb_dim
        }
    }

    # dense features ----------------------------------
    feature_list_dense=[
        {
            'feature_name':'item_dense_feature',
            'feature_dim':40,
            'emb_dim':40
        },
        {
            'feature_name':'user_dense_feature',
            'feature_dim':23,
            'emb_dim':23
        }
    ]

    full_feature_dims_dense = []
    full_feature_names_dense = []
    full_emb_dim_dense = []

    for e in feature_list_dense:
        full_feature_names_dense.append(e['feature_name'])
        full_feature_dims_dense.append(e['feature_dim'])
        full_emb_dim_dense.append(e['emb_dim'])

    full_feature_names_dense = (full_feature_names_dense)

    print('full_feature_names_dense',full_feature_names_dense)
    print('full_feature_dims_dense',full_feature_dims_dense)

    full_emb_dim_sum += np.sum(full_emb_dim_dense)
    #----------------------------------------------------
 

    neg_num = 6

    type_cnt = 2
    
    # TODO
    node_id = tensors[0]
    source = tensors[1]
    base_runner.add_trace_variable("node_id_str", node_id)
    att_sim_list = []

    feature_dic,src_dense = sample_pos_n_feature_inf(source,full_feature_names,part_q_feature_names,part_i_feature_names,part_c_feature_names,full_feature_names_dense,full_feature_dims_dense)
    full_node_dense = [src_dense ]

    with tf.variable_scope("finetune_embedding", reuse=tf.AUTO_REUSE) as scope:
        embedding_dic = {}
        for e_part in feature_dic:
            node_embed = sparse_embedding(feature_dic[e_part], dic_info[e_part]['feature_dims'],
                                               dic_info[e_part]['emb_dim'],
                                               ps_num,
                                               init_val=embedding_ini_val,
                                               stddev=embedding_stddev,
                                               names=dic_info[e_part]['feature_names'])
            node_embed_c = tf.concat(node_embed, 1)

            if e_part == 'self':
                full_node_embed_dense = dense_embedding(full_node_dense[0], full_feature_dims_dense,
                                        full_emb_dim_dense,
                                        ps_num,
                                        init_val=dense_ini_val,
                                        stddev=dense_stddev,
                                        names=full_feature_names_dense)                                   
                full_node_embed_dense_c = tf.concat(full_node_embed_dense, 1)
                node_embed_c = tf.concat([node_embed_c,full_node_embed_dense_c], 1)

            embedding_dic[e_part] = node_embed_c
         
         

    src_input = fake_convolution(embedding_dic['self'], embedding_dic['q'], embedding_dic['i'],embedding_dic['c'],
                q_dim=part_q_emb_dim_sum,i_dim=part_i_emb_dim_sum,c_dim=part_c_emb_dim_sum)
 

    with tf.variable_scope("finetune_main_current_dnn", reuse=tf.AUTO_REUSE):
        current_id_deep = id_dnn_net(src_input,
                                     full_emb_dim_sum, ps_num,
                                     dense_ini_val,
                                     dense_stddev)
    with tf.variable_scope("finetune_main_current_ays", reuse=tf.AUTO_REUSE):
        current_id_ays = id_ays_net(current_id_deep, ps_num, dense_ini_val, dense_stddev)
        base_runner.add_trace_variable("cur_embedding", current_id_ays)

    with tf.variable_scope("finetune_main_node_dnn", reuse=tf.AUTO_REUSE) as scope:
        # node_neg_id_deep = id_dnn_net(neg_input_all, emb_feature_dim * 11, ps_num,
        #                               dense_ini_val,
        #                               dense_stddev)
        # scope.reuse_variables()
        pos_id_deep = id_dnn_net(src_input, full_emb_dim_sum, ps_num,
                                 dense_ini_val,
                                 dense_stddev)

    with tf.variable_scope("finetune_main_node_ays", reuse=tf.AUTO_REUSE) as scope:
        # node_neg_id_ays = id_ays_net(node_neg_id_deep, ps_num, dense_ini_val, dense_stddev)
        # scope.reuse_variables()
        pos_id_ays = id_ays_net(pos_id_deep, ps_num, dense_ini_val, dense_stddev)
        base_runner.add_trace_variable("pos_embedding", pos_id_ays)

    with tf.variable_scope("finetune_att_sim") as scope:
        pos_sim = cosine_fun(current_id_ays, pos_id_ays)
        att_sim = [pos_sim]
        # need to be modified
        # node_neg_id_ays_re = tf.reshape(node_neg_id_ays, [-1, neg_num * 128])
        # node_neg_id_ays_list = tf.split(node_neg_id_ays_re, num_or_size_splits=neg_num, axis=1)
        # for neg in node_neg_id_ays_list:
        #     neg_sim = cosine_fun(current_id_ays, neg)
        #     att_sim.append(neg_sim)

    att_sim_con = tf.concat(att_sim, 1)
    att_sim_list.append(att_sim_con)
    return tf.concat(att_sim_list, 0)


def id_dnn_net(id_embedding, input_dim, ps_num, init_val, stddev):
    with tf.variable_scope("layer1"):
        layer1_output0 = full_connect(id_embedding, [input_dim, 256], [256], False, ps_num, init_val, stddev)
        layer1_output = tf.nn.elu(layer1_output0)

    with tf.variable_scope("layer2"):
        layer2_output0 = full_connect(layer1_output, [256, 256], [256], False, ps_num, init_val, stddev)
        layer2_output = tf.nn.elu(layer2_output0)

    return layer2_output


def id_ays_net(id_deep, ps_num, init_val, stddev):
    with tf.variable_scope("layer1"):
        x = full_connect(id_deep, [256, 32], [32], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        x=tf.nn.l2_normalize(x,axis=1)
    return x


def cosine_fun(ays_src, ays_dst):
    src_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_src), 1, True))
    dst_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_dst), 1, True))

    prod = tf.reduce_sum(tf.multiply(ays_src, ays_dst), 1, True)
    norm_prod = tf.multiply(src_norm, dst_norm)

    cosine = tf.truediv(prod, norm_prod)
    return cosine


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

def full_connect_sparse(train_inputs, weights_shape, biases_shape, no_biases, ps_num, sp_weights, init_val, stddev):
    # weights
    from tf_ps.ps_context import variable_info
    with variable_info(batch_read=3000, var_type='hash'):
        if ps_num > 1:
            weights = tf.get_variable("weights",
                                      weights_shape,
                                      initializer=get_initializer(init_val=init_val, stddev=stddev),
                                      partitioner=tf.min_max_variable_partitioner(max_partitions=ps_num))
        else:
            weights = tf.get_variable("weights",
                                      weights_shape,
                                      initializer=get_initializer(init_val=init_val, stddev=stddev))

    train = tf.nn.embedding_lookup_sparse(weights, sp_ids=train_inputs, sp_weights=sp_weights, combiner="sum")
    if not no_biases:
        # biases
        biases = tf.get_variable("biases",
                                 biases_shape,
                                 initializer=get_initializer(init_val=init_val, stddev=stddev))
        train = train + biases
    return train


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


def softmax_loss(sim, weight_decay=0.0001, gama=5.0):
    prob = tf.nn.softmax(gama * sim)
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.log(hit_prob)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return tf.reduce_mean(loss, name='ranking_loss') + weight_decay * tf.add_n(reg_losses)

def auc(sim, num_thresholds=200, decay_rate=1):
    neg_num = tf.shape(sim)[1] - 1
    sample_num = tf.shape(sim)[0]
    labels_matrix = tf.concat([tf.ones([sample_num, 1], tf.int32), tf.zeros([sample_num, neg_num], tf.int32)], axis=1)
    labels = tf.reshape(labels_matrix, [-1, 1])

    predictions = tf.reshape(tf.nn.sigmoid(sim, name='sigmoid_auc'), [-1, 1])
    _, auc_op = tf.contrib.metrics.streaming_auc(predictions, labels, num_thresholds=num_thresholds,
                                                 decay_rate=decay_rate)

    return auc_op