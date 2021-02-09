from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tf_euler

import tensorflow as tf
from tensorflow.python.util import nest
import tf_context as ctx
import base_runner
import graph_tag
import numpy as np

from pareto_mtl import ParetoMTL
import tf_sort

 

nation_sp = ctx.get_config("nation")
 

print('nation',nation_sp)
# print('flag_exp_loss',flag_exp_loss)
gm_loss_weight = 36.
flag_dist = 1.
flag_RI = 0
var_weight=1e-5
gate_weight = 1.
gate_th=0.8
flag_gate_above=True
same_nation_weight=0.0
flag_uniform_nation_weight=False
cate_align_weight=0.1
 
print('gm_loss_weight',gm_loss_weight)
print('flag_dist',flag_dist)
print('flag_RI',flag_RI)
print('var_weight',var_weight)
print('gate_weight',gate_weight)
print('gate_th',gate_th)
print('flag_gate_above',flag_gate_above)
print('same_nation_weight',same_nation_weight)
print('flag_uniform_nation_weight',flag_uniform_nation_weight)
print('cate_align_weight',cate_align_weight)
 
if flag_RI:
    from util_pretrain_RI import *
else:
    from util_pretrain import *

if flag_dist:
    from graph_matching_pretrain_dist import *
else:
    from graph_matching_pretrain_prod import *
 


def greater_than_zero(x):
    x=tf.maximum(0.0*x,x)
    x = tf.sign(x)
    return x


 
 

def random_randint(limit,n_row):
    a = tf.random_uniform((n_row,limit))
    a = tf.argmax(a,axis=1)
    return a

 


def get_grad_norm(grad_r):
    grad_norm = 0.0
    for i_grad_r,e_grad_r in enumerate(grad_r):
        if e_grad_r is None:
            break
        else:
            grad_norm += tf.reduce_sum(tf.square(e_grad_r))
    grad_norm = tf.maximum(0.0*grad_norm + 1e-7,grad_norm)
    grad_norm = tf.sqrt(grad_norm)
    return grad_norm

def get_grad_prod(grad_1,grad_2):
    grad_prod = 0.0
    for i in range(len(grad_1)):
        if grad_1[i] is None or grad_2[i] is None :
            break
        else:
            grad_prod += tf.reduce_sum(grad_1[i]*grad_2[i])
    
    return grad_prod
 

def stop_grad_list(x_list):
  
    return [tf.stop_gradient(x) for x in x_list]

def gen_paired_meta_loss(main_loss,aux_loss, weights, flag_stop_gradient):
 
    lm,ls = main_loss,aux_loss 
   
    grad_m = tf.gradients(lm, weights)
    grad_norm_m = get_grad_norm(grad_m)
    grad_s = tf.gradients(ls, weights)
    grad_norm_s = get_grad_norm(grad_s)
     
    if flag_stop_gradient:
        grad_m = stop_grad_list(grad_m)
        grad_norm_m = tf.stop_gradient(grad_norm_m)
  
    loss_grad_match = 0.
  
    grad_prod = get_grad_prod(grad_m, grad_s)
    loss_grad_match += - grad_prod / (grad_norm_m * grad_norm_s)
      
    return tf.reduce_mean(loss_grad_match)

def get_variable_list(regex_pattern):
        import re
#         regex_pattern = 'hidden[12]'
        train_vars = []
        var_names = []
        size_list = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if re.search(regex_pattern, var.op.name):
                train_vars.append(var)
                var_names.append(var.op.name)
                size_list.append(tf.shape(var))
        print('----------------------------- Begin: var_names of %s -----------------------------'%regex_pattern)
        print(var_names)
        print('----------------------------- End: var_names of %s -----------------------------'%regex_pattern)
        
        return train_vars,size_list

def grad_flatten(grads,size_list):
    grad_list = []
    for i,e in enumerate(grads):
        if e is not None:
            x =e 
        else:
            x = tf.zeros(size_list[i])
        grad_list.append(tf.reshape(x,[-1]))
    return tf.concat(grad_list,0)


# from epo_lp import EPO_LP

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
    print('len(sp_tensors)',len(sp_tensors))
    print('len(names)',len(names))
    for i in range(len(sp_tensors)):
        with tf.variable_scope(names[i]):
            print("sparse_embedding name:" + names[i] + " dim:" + str(input_dimensions[i]))
            rst = full_connect_sparse(sp_tensors[i], [input_dimensions[i], embedding_dimension[i]], None, True, ps_num,
                                      None, init_val, stddev)
            l.append(rst)
    return l

def init_graph():
    zk_addr = ctx.get_config('euler', 'zk_addr')
    zk_path = '/euler/{}'.format(ctx.get_app_id())
    shard_num = ctx.get_config('euler', 'shard_num')
    tf_euler.initialize_graph({
          'mode': 'remote',
          'zk_server': zk_addr,
          'zk_path': zk_path,
          'shard_num': shard_num,
          'num_retries':10
      })

def sample_by_walk(src, edge_type, walk_len=1, walk_p=1, walk_q=1):
    walk_path = [edge_type]
    walk_path.extend([['1'] for i in range(walk_len - 1)])
    path = tf_euler.random_walk(
        src, walk_path,
        p=walk_p,
        q=walk_q)

    path_s = tf.slice(path, [0,1], [-1, walk_len])
    pos = tf.reshape(path_s, [-1])
    negs_copy = tf_euler.sample_node_with_src(pos, 6 * walk_len)

    src_s = tf.reshape(src, [-1, 1])
    src_copy = tf.concat([src_s for i in range(walk_len)], 1)
    #have some problems
    neg_nodes = tf.reshape(negs_copy, [-1])
    src_nodes = tf.reshape(src_copy, [-1])
    return src_nodes, pos, neg_nodes


def get_neighbors(src, pos, negs,q_part_features,i_part_features, i_nei_cnt=5, q_nei_cnt=5):
    # q_part_features = ['query_norm']
    # q_features = "query_sorted_norm_keyword"

    # i_part_features = ['item_id']

    src_i_nodes, _, _ = tf_euler.sample_neighbor(src, edge_types=['1', '3'], count=i_nei_cnt)
    print("======== neighbor count: {}".format(i_nei_cnt))
    src_i_nodes_filled = tf_euler.get_sparse_feature(src_i_nodes, i_part_features)
    src_q_nodes, _, _ = tf_euler.sample_neighbor(src, edge_types=['2'], count=q_nei_cnt)
    src_q_nodes_filled = tf_euler.get_sparse_feature(src_q_nodes, q_part_features)

    pos_i_nodes, _, _ = tf_euler.sample_neighbor(pos, edge_types=['1', '3'], count=i_nei_cnt)
    pos_i_nodes_filled = tf_euler.get_sparse_feature(pos_i_nodes, i_part_features)
    pos_q_nodes, _, _ = tf_euler.sample_neighbor(pos, edge_types=['2'], count=q_nei_cnt)
    pos_q_nodes_filled = tf_euler.get_sparse_feature(pos_q_nodes, q_part_features)

    neg_i_nodes, _, _ = tf_euler.sample_neighbor(negs, edge_types=['1', '3'], count=i_nei_cnt)
    neg_i_nodes_filled = tf_euler.get_sparse_feature(neg_i_nodes, i_part_features)
    neg_q_nodes, _, _ = tf_euler.sample_neighbor(negs, edge_types=['2'], count=q_nei_cnt)
    neg_q_nodes_filled = tf_euler.get_sparse_feature(neg_q_nodes, q_part_features)

    tohash128(src_i_nodes_filled)
    tohash128(src_q_nodes_filled)
    tohash128(pos_i_nodes_filled)
    tohash128(pos_q_nodes_filled)
    tohash128(neg_i_nodes_filled)
    tohash128(neg_q_nodes_filled)

    return src_i_nodes_filled, src_q_nodes_filled, pos_i_nodes_filled, \
           pos_q_nodes_filled, neg_i_nodes_filled, neg_q_nodes_filled

def get_feature(src,full_features):
    src_filled = tf_euler.get_sparse_feature(src, full_features)
    tohash128(src_filled)
    return src_filled

def get_neighbor(x,q_part_features,i_part_features,c_part_features, q_nei_cnt=5,i_nei_cnt=5,c_nei_cnt=5 ):
 

    q_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['2','5'], count=i_nei_cnt)
    i_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['0','1', '3','7'], count=i_nei_cnt)
    c_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['46','66', '9'], count=i_nei_cnt)
    
#     q_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['2' ], count=i_nei_cnt)
#     i_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['1', '3' ], count=i_nei_cnt)
#     c_nodes, _, _ = tf_euler.sample_neighbor(x, edge_types=['46','66', '9'], count=i_nei_cnt)
    
     
    q_f = get_feature(q_nodes,q_part_features)
    i_f = get_feature(i_nodes,i_part_features)
    c_f = get_feature(c_nodes,c_part_features)
    
    return q_f,i_f,c_f
     
def sample_pos_n_feature(source,full_features,q_part_features,i_part_features,c_part_features):
    node_list = []
    src, pos, neg = sample_by_walk(source, ['0','1', '3'])
    node_list.append({'src':src,'pos':pos,'neg':neg})
    
    for e_level in ['6','5','4','3','2','1']:
        _, pos, neg = sample_by_walk(source, ['4'+e_level, '6'+e_level])
        node_list.append({'src':src,'pos':pos,'neg':neg})
        
    feature_dic = {'self':[],'q':[],'i':[],'c':[]}
    for e in node_list:
        dic_self,dic_q,dic_i,dic_c ={},{},{},{}
        for e_type in e:
            self_f = get_feature(e[e_type],full_features)
            q_f,i_f,c_f = get_neighbor(e[e_type],q_part_features,i_part_features,c_part_features)
            dic_self[e_type],dic_q[e_type],dic_i[e_type],dic_c[e_type]=self_f,q_f,i_f,c_f
        feature_dic['self'].append(dic_self)
        feature_dic['q'].append(dic_q)
        feature_dic['i'].append(dic_i)
        feature_dic['c'].append(dic_c)
            
         
    return feature_dic


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
                   
      
     
    nation_list=['TH','VN','SG','MY','ID','PH']
    feature_list=[
        {
            'feature_name':'query_feature1',
            'feature_dim':int(182309924*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'query_feature2',
            'feature_dim':int(13286936*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature1',
            'feature_dim':int(18357245*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_id',
            'feature_dim':int(405929642*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'global_category_id',
            'feature_dim':int(4273*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature9',
            'feature_dim':int(175863*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature10',
            'feature_dim':int(1025024*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature11',
            'feature_dim':int(461654*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature12',
            'feature_dim':int(1030492*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature13',
            'feature_dim':3,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature14',
            'feature_dim':9,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature15',
            'feature_dim':4,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature16',
            'feature_dim':9,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature17',
            'feature_dim':7,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature18',
            'feature_dim':3,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature2',
            'feature_dim':int(10683948*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature3',
            'feature_dim':int(21677772*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature5',
            'feature_dim':int(117184*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature6',
            'feature_dim':int(687583*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature7',
            'feature_dim':int(339428*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'category_feature2',
            'feature_dim':int(3425*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'query_feature3',
            'feature_dim':int(14318897*1.1),
            'emb_dim':8
        },
        {
            'feature_name':'item_feature19',
            'feature_dim':2,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature10',
            'feature_dim':11,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature11',
            'feature_dim':4,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature12',
            'feature_dim':4,
            'emb_dim':8
        },
        {
            'feature_name':'item_feature13',
            'feature_dim':2,
            'emb_dim':8
        },
        {
            'feature_name':'level',
            'feature_dim':7,
            'emb_dim':8
        },
        {
            'feature_name':'is_leaf',
            'feature_dim':2,
            'emb_dim':8
        },
        {
            'feature_name':'nation',
            'feature_dim':6,
            'emb_dim':8
        }
    ]

 
    # dic = {
    #     'ID':	6191,
    #     'MY':	4632,
    #     'PH':	6036,
    #     'SG':	4224,
    #     'TH':	7153,
    #     'VN':	4115
    # }
    # for e_nation in nation_list:
    #     feature_list.append(
    #         {
    #             'feature_name':'%s_venture_category_name_l10n_seg' % e_nation,
    #             'feature_dim':int(dic[e_nation]*1.1),
    #             'emb_dim':8
    #         }
    #     )
 
    dic = {
        '1':	53,
        '2':	361,
        '3':	1614,
        '4':	2159,
        '5':	7033,
        '6':	4380
    }
    for e_level in ['1','2','3','4','5','6']:
        feature_list.append(
            {
                'feature_name':'global_category_id_level%s' % e_level,
                'feature_dim':int(dic[e_level]*1.1),
                'emb_dim':8
            }
        )

    dic = {
        '6':	276,
        '7':	2708,
        '8':	4791,
        '9':	5432,
        '10':	3903,
        '11':	2534
    }
    
    for e_level in ['6','7','8','9','10','11']:
        feature_list.append(
            {
                'feature_name':'category_feature%s' % e_level,
                'feature_dim':int(dic[e_level]*1.1),
                'emb_dim':8
            }
        )

    dic = {
        '12':	569,
        '13':	3985,
        '14':	9136,
        '15':	11316,
        '16':	8008,
        '17':	4697
    }
    for e_level in ['12','13','14','15','16','17']:
        feature_list.append(
            {
                'feature_name':'category_feature%s' % e_level,
                'feature_dim':int(dic[e_level]*1.1),
                'emb_dim':8
            }
        )
                   
      
 
    # dic = {
    #     'ID':	6192,
    #     'MY':	4632,
    #     'PH':	6036,
    #     'SG':	4224,
    #     'TH':	7153,
    #     'VN':	4115
    # }
    # for e_nation in nation_list:
    #     feature_list.append(
    #         {
    #             'feature_name':'%s_venture_category_name_l10n_seg_alllevel' % e_nation,
    #             'feature_dim':int(dic[e_nation]*1.1),
    #             'emb_dim':8
    #         }
    #     )
   
    full_feature_dims = []
    full_feature_names = []
    full_emb_dim = []

    for e in feature_list:
        full_feature_names.append(e['feature_name'])
        full_feature_dims.append(e['feature_dim'])
        full_emb_dim.append(e['emb_dim'])

    full_feature_names = (full_feature_names)
 
    part_q_feature_names = ['query_norm' ]
    part_q_feature_dims = [ int(182309924*1.1) ]
    part_q_emb_dim = [8 ]

    part_i_feature_names = ['item_id' ]
    part_i_feature_dims = [int(405929642*1.1) ]
    part_i_emb_dim = [8 ]
    
    part_c_feature_names = ['venture_category_name_en' ]
    part_c_feature_dims = [int(4273*1.1) ]
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
 
    
    neg_num = 6
    walk_len=1

    type_cnt = 2

    swd_dim = 128
    swd_dim_gwd = 8
    
    swd_limit = 512
    swd_limit_gwd = 128

    # model_config = ctx.get_config('reader')
    batch_size = ctx.get_config('batch_size')

    init_graph()

    source=tensors.data[0][0].values
    print('len(tensors.data[0])',len(tensors.data[0]))
    print('len(tensors.data)',len(tensors.data))
    # print('source[:5]',source[:5])
    print('tensors.data[1]',tensors.data[1])
    print('tensors.data[2]',tensors.data[2])
    print('tensors.data[3]',tensors.data[3])
    print('tensors.data[4]',tensors.data[4])
    # print('tensors.data[0][1].values[:5]',tensors.data[0][1].values[:5])
    # print('tensors.data[0][2].values[:5]',tensors.data[0][2].values[:5])
    # negs = tensors.data[1][0].values

    
    
    print('tf.__version__', tf.__version__)

    import sys
    print('sys.version',sys.version)

    dic_nation2int={'TH':1,'VN':2,'SG':3,'MY':4,'ID':5,'PH':6}
    nation_vec = tensors.data[1]
    nation_onehot = []
    nation_idx_where = []
  
    for e_nation in nation_list:
        eq_nation = tf.to_float(tf.equal(tf.cast(nation_vec,tf.int32), tf.cast(dic_nation2int[e_nation],tf.int32)))
        nation_onehot.append(eq_nation)
        idx_where = tf.cast(tf.where(tf.equal(eq_nation,1))[:,0],dtype=tf.int32)
        idx_where = tf.concat([idx_where,[tf.shape(eq_nation)[0],tf.shape(eq_nation)[0]+1]],0)
        nation_idx_where.append(idx_where)
    
    nation_onehot = tf.concat(nation_onehot,1)
    nation_exist = tf.to_float(tf.greater(tf.reduce_sum(nation_onehot,0),0)) 
    
    wd_idx = tf.range(swd_limit)
    wd_idx_norm = tf.to_float(wd_idx)/tf.to_float(swd_limit)
    
    gwd_idx_left = random_randint(swd_limit_gwd,swd_limit_gwd)
    gwd_idx_right = random_randint(swd_limit_gwd,swd_limit_gwd)
    gwd_idx = [gwd_idx_left,gwd_idx_right]
    gwd_idx_norm = [tf.to_float(gwd_idx_left)/tf.to_float(swd_limit),
                    tf.to_float(gwd_idx_right)/tf.to_float(swd_limit)]
    
    cgwd_idx_left = random_randint(swd_limit_gwd,swd_limit_gwd*(walk_len+neg_num))
    cgwd_idx_right = random_randint(swd_limit_gwd*(walk_len+neg_num),swd_limit_gwd*(walk_len+neg_num))
    cgwd_idx = [cgwd_idx_left,cgwd_idx_right]
    cgwd_idx_norm = [tf.to_float(cgwd_idx_left)/tf.to_float(swd_limit),
                    tf.to_float(cgwd_idx_right)/tf.to_float(swd_limit*(walk_len+neg_num))]
    

    Theta=tf.random_normal((swd_dim,32))
    Theta=tf.nn.l2_normalize(Theta,axis=1)
    
    Theta_gwd=tf.random_normal((swd_dim_gwd,32))
    Theta_gwd=tf.nn.l2_normalize(Theta_gwd,axis=1)
    

    source = get_high64bit(source)
    # negs = get_high64bit(negs)

    

    feature_dic = sample_pos_n_feature(source,full_feature_names,part_q_feature_names,part_i_feature_names,part_c_feature_names)

    # contain_same_cate ------------
    cate_vec = tensors.data[4]
    cate_vec = tf.to_float(cate_vec)
    cate_vec = tf.reshape(cate_vec,[-1,1])
    n_nation=len(nation_list)
    cate_mat = nation_onehot * cate_vec
    contain_same_cate = []
    for i_nation in range(n_nation):
        i_cate = tf.reshape(cate_mat[:,i_nation],[-1,1])
        i_mask = tf.reshape(nation_onehot[:,i_nation],[-1,1])
        for j_nation in range(n_nation):
            j_cate = tf.reshape(cate_mat[:,j_nation],[-1,1])
            j_mask = tf.reshape(nation_onehot[:,j_nation],[-1,1])
            ij_mask = tf.matmul(i_mask,j_mask,transpose_b=True)
            cate_join = tf.to_float(tf.equal(i_cate, tf.transpose(j_cate))) * ij_mask
            cate_join = tf.reduce_sum(cate_join,keepdims=True)
            contain_same_cate.append(cate_join)
    contain_same_cate = tf.concat(contain_same_cate,0)
    contain_same_cate = tf.reshape(contain_same_cate,[n_nation,n_nation])
    contain_same_cate = greater_than_zero(contain_same_cate)
    #--------------------------------

    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE) as scope:
        embedding_dic={'self':[],'q':[],'i':[],'c':[]}
        for e_part in feature_dic:
            e_list = feature_dic[e_part]
            for e in e_list:
                dic={}
                for e_type in e:
                    node_embed = sparse_embedding(e[e_type], dic_info[e_part]['feature_dims'],
                                               dic_info[e_part]['emb_dim'],
                                               ps_num,
                                               init_val=embedding_ini_val,
                                               stddev=embedding_stddev,
                                               names=dic_info[e_part]['feature_names'])
                    node_embed_c = tf.concat(node_embed, 1)
                    dic[e_type]=node_embed_c
                embedding_dic[e_part].append(dic)
    
    input_list = []        
    n_level = len(embedding_dic['self'])
    for i_level in range(n_level):
        dic={}
        for e_type in embedding_dic['self'][i_level]:
            e_input = fake_convolution(embedding_dic['self'][i_level][e_type],
                                       embedding_dic['q'][i_level][e_type],
                                       embedding_dic['i'][i_level][e_type],
                                       embedding_dic['c'][i_level][e_type],
                    q_dim=part_q_emb_dim_sum,i_dim=part_i_emb_dim_sum,c_dim=part_c_emb_dim_sum)
            dic[e_type]=e_input
        input_list.append(dic)

     
        
    with tf.variable_scope("gate_current_dnn", reuse=tf.AUTO_REUSE):
        gate_current_id_deep_list = []
        gate_current_id_deep_list_swd=[]
        current_id_ays_list_swd=[]
        current_id_ays_list_sgwd=[]
        for i_dic,e_dic in enumerate(input_list):
            current_id_deep = gate_feature_net(e_dic['src'],
                                         full_emb_dim_sum, ps_num,
                                         dense_ini_val,
                                         dense_stddev)
            current_id_deep_gate = gate_wd_net(current_id_deep, ps_num, dense_ini_val, dense_stddev)
            gate_current_id_deep_list.append(current_id_deep)
            
            if flag_RI:
                current_id_deep_for_sort = mapping_fun(current_id_deep, ps_num, dense_ini_val, dense_stddev,nation_list)
            else:
                current_id_deep_for_sort = current_id_deep
            x_list_wd,idx_list_wd=swd_proj_src(current_id_deep_for_sort, Theta,nation_idx_where)
            x_list_gwd,idx_list_gwd=swd_proj_src(current_id_deep_for_sort, Theta_gwd,nation_idx_where)
             
            gate_feature_list=swd_proj_src_with_idx(current_id_deep, idx_list_gwd,nation_idx_where)
            gate_list=swd_proj_src_with_idx(current_id_deep_gate, idx_list_wd,nation_idx_where)
            gate_current_id_deep_list_swd.append([gate_feature_list,gate_list])
            
            current_id_ays_list_swd.append([x_list_wd,idx_list_wd])
            current_id_ays_list_sgwd.append([x_list_gwd,idx_list_gwd])
             
        
    with tf.variable_scope("gate_node_dnn", reuse=tf.AUTO_REUSE) as scope:
        gate_pos_id_deep_list,gate_node_neg_id_deep_list = [],[]
        gate_node_id_ays_list_swd=[]
        node_id_ays_list_swd = []
        for i_dic,e_dic in enumerate(input_list):
            pos_id_deep = gate_feature_net(e_dic['pos'], full_emb_dim_sum, ps_num,
                                     dense_ini_val,
                                     dense_stddev)
            node_neg_id_deep = gate_feature_net(e_dic['neg'], full_emb_dim_sum, ps_num,
                                          dense_ini_val,
                                          dense_stddev)
            gate_pos_id_deep_list.append(pos_id_deep)
            gate_node_neg_id_deep_list.append(node_neg_id_deep)
            
            pos_id_deep_gate = gate_wd_net(pos_id_deep, ps_num, dense_ini_val, dense_stddev)
            node_neg_id_deep_gate = gate_wd_net(node_neg_id_deep, ps_num, dense_ini_val, dense_stddev)
            
            if flag_RI:
                pos_id_deep_for_sort = mapping_fun(pos_id_deep, ps_num, dense_ini_val, dense_stddev,nation_list)
                node_neg_id_deep_for_sort = mapping_fun(node_neg_id_deep, ps_num, dense_ini_val, dense_stddev,nation_list)
            else:
                pos_id_deep_for_sort = pos_id_deep
                node_neg_id_deep_for_sort = node_neg_id_deep

            
            x_list_gwd,idx_list_gwd=swd_proj_pos_neg(pos_id_deep_for_sort,node_neg_id_deep_for_sort, Theta_gwd, nation_idx_where,neg_num)
  
            gate_feature_list=swd_proj_pos_neg_with_idx(pos_id_deep,node_neg_id_deep, idx_list_gwd, nation_idx_where,neg_num)
            gate_list=swd_proj_pos_neg_with_idx(pos_id_deep_gate,node_neg_id_deep_gate, idx_list_gwd, nation_idx_where,neg_num)
            gate_node_id_ays_list_swd.append([gate_feature_list,gate_list])
            
            node_id_ays_list_swd.append([x_list_gwd,idx_list_gwd])
             
        
        #for inference
        gate_node_src_context_ays_swd = []
        node_src_context_ays_swd = []
        node_src_context_ays_sgwd = []
        gate_node_src_context_deep = gate_feature_net(input_list[0]['src'], full_emb_dim_sum, ps_num,
                                      dense_ini_val,
                                      dense_stddev)
        gate_node_src_context_deep_gate = gate_wd_net(gate_node_src_context_deep, ps_num, dense_ini_val, dense_stddev)
        
        if flag_RI:
            gate_node_src_context_deep_for_sort = mapping_fun(gate_node_src_context_deep, ps_num, dense_ini_val, dense_stddev,nation_list)
        else:
            gate_node_src_context_deep_for_sort = gate_node_src_context_deep

        x_list_wd,idx_list_wd=swd_proj_src(gate_node_src_context_deep_for_sort, Theta,nation_idx_where)
        x_list_gwd,idx_list_gwd=swd_proj_src(gate_node_src_context_deep_for_sort, Theta_gwd,nation_idx_where)
 
        gate_feature_list=swd_proj_src_with_idx(gate_node_src_context_deep, idx_list_gwd,nation_idx_where)
        gate_list=swd_proj_src_with_idx(gate_node_src_context_deep_gate, idx_list_wd,nation_idx_where)
        gate_node_src_context_ays_swd.append([gate_feature_list,gate_list])
        
        node_src_context_ays_swd.append([x_list_wd,idx_list_wd])
        node_src_context_ays_sgwd.append([x_list_gwd,idx_list_gwd])
        
          
    with tf.variable_scope("gate_att_sim", reuse=tf.AUTO_REUSE) as scope:
        n_level = len(gate_current_id_deep_list)
        gate_att_sim_list = []
        for i_level in range(n_level):
            current_id_ays=gate_current_id_deep_list[i_level]
            pos_id_ays = gate_pos_id_deep_list[i_level]
            node_neg_id_ays = gate_node_neg_id_deep_list[i_level]
            
            current_id_ays = tf.reshape(current_id_ays, [-1, 1, 32])
            pos_id_ays = tf.reshape(pos_id_ays, [-1, walk_len, 32])
            node_neg_id_ays_re = tf.reshape(node_neg_id_ays, [-1, neg_num, 32])
            node_id_ays = tf.concat([pos_id_ays, node_neg_id_ays_re], 1)
            att_sim_con = batch_cosine_fun(current_id_ays, node_id_ays)
            gate_att_sim_list.append(att_sim_con)
        
        att_sim_list = gate_att_sim_list

    print('tensors.data[1]',tensors.data[1])
    print('tensors.data[2]',tensors.data[2])
    print('tensors.data[3]',tensors.data[3])
    print('tensors.data[4]',tensors.data[4])
    print('tf.__version__', tf.__version__)
    import sys
    print('sys.version',sys.version)
    
    # n_tasks,n_params,preference = n_level,1,np.array([1,0,0,0,0,0,0])
    # epo_lp = EPO_LP(m=n_tasks, n=n_params, r=preference)

    align_info = [current_id_ays_list_swd,node_id_ays_list_swd,node_src_context_ays_swd,\
                  current_id_ays_list_sgwd,node_src_context_ays_sgwd,\
                      wd_idx,wd_idx_norm,gwd_idx,gwd_idx_norm,cgwd_idx,cgwd_idx_norm,nation_exist,\
                  gate_current_id_deep_list_swd,gate_node_id_ays_list_swd,gate_node_src_context_ays_swd,\
                 ps_num, dense_ini_val, dense_stddev,\
                 gate_att_sim_list,contain_same_cate]

    
    return att_sim_list,nation_onehot,align_info


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

def gate_feature_net(id_embedding, input_dim, ps_num, init_val, stddev):
    with tf.variable_scope("gate_feature_layer1"):
        x = full_connect(id_embedding, [input_dim, 256], [256], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)

    with tf.variable_scope("gate_feature_layer2"):
        x = full_connect(x, [256, 256], [256], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        
    with tf.variable_scope("gate_feature_layer3"):
        x = full_connect(x, [256, 32], [32], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        x=tf.nn.l2_normalize(x,axis=1)
    return x

def gate_wd_net(id_deep, ps_num, init_val, stddev):
    with tf.variable_scope("wd_layer1"):
        x = full_connect(tf.stop_gradient(id_deep), [32, 6], [6], False, ps_num, init_val, stddev)
        x = tf.sigmoid(x)
    return x

def gate_gwd_net(id_deep, ps_num, init_val, stddev, sp_name):
    with tf.variable_scope("gwd_layer1_%s"%sp_name):
        x = full_connect(tf.stop_gradient(id_deep), [96, 32], [32], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        
    with tf.variable_scope("gwd_layer2_%s"%sp_name):
        x = full_connect(x, [32, 6], [6], False, ps_num, init_val, stddev)
        x = tf.sigmoid(x)
         
    return x

def gate_cgwd_net(id_deep, ps_num, init_val, stddev, sp_name):
    with tf.variable_scope("cgwd_layer1_%s"%sp_name):
        x = full_connect(tf.stop_gradient(id_deep), [96, 32], [32], False, ps_num, init_val, stddev)
        x = tf.nn.elu(x)
        
    with tf.variable_scope("cgwd_layer2_%s"%sp_name):
        x = full_connect(x, [32, 6], [6], False, ps_num, init_val, stddev)
        x = tf.sigmoid(x)
         
    return x

def batch_cosine_fun(ays_src, ays_dst):
    # ays_src [batch, 1, dim]
    # ays_dst [batch, num, dim]
    src_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_src), 2, True)) # [batch, 1, 1]
    src_norm = tf.squeeze(src_norm, -1) # [batch, 1]
    dst_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_dst), 2, True)) # [batch, num, 1]
    dst_norm = tf.squeeze(dst_norm, -1) # [batch, num]
    
    prod = tf.matmul(ays_src, ays_dst, transpose_b=True) # [batch, 1, num]
    prod = tf.squeeze(prod, 1) # [batch, num]
    norm_prod = src_norm * dst_norm # [batch, num]
    cosine = tf.truediv(prod, norm_prod)
    return cosine

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

    train = tf.nn.embedding_lookup_sparse(weights, sp_ids=train_inputs, sp_weights=sp_weights, combiner="sum") #"mean", "sqrtn" and "sum"
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


def sigmoid_loss(input_list, weight_decay=0.0001, gama=5.0):
#     sim_list,epo_lp = input_list
#     m = len(sim_list)
    
    sim_list,nation_onehot,align_info  = input_list
    
    current_id_ays_list_swd,node_id_ays_list_swd,node_src_context_ays_swd,\
    current_id_ays_list_sgwd,node_src_context_ays_sgwd,\
    wd_idx,wd_idx_norm,gwd_idx,gwd_idx_norm,cgwd_idx,cgwd_idx_norm,nation_exist,\
    gate_current_id_deep_list_swd,gate_node_id_ays_list_swd,gate_node_src_context_ays_swd,\
    ps_num, init_val, stddev,\
    gate_att_sim_list,contain_same_cate= align_info
    
      
    
    

    
    nation_mean = nation_onehot/(tf.reduce_sum(nation_onehot, 0, keep_dims=True)+1e-6)
    
    solver=ParetoMTL()

    
     
    loss_sum = 0.0
    cont_loss = 0.0
    for i,sim in enumerate(sim_list):
        if i > 1 and i < len(sim_list)-1:
            continue
        one_labels = tf.ones([tf.shape(sim)[0], 1], dtype=tf.float32)
        zero_labels = tf.zeros([tf.shape(sim)[0], 6], dtype=tf.float32)
        label = tf.concat([one_labels, zero_labels], -1)

        if i==0:
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=gama*sim, pos_weight=2.0)
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gama*sim, labels=label)
        else:
            # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gama*sim, labels=label)
            prob = tf.nn.softmax(gama * sim)
            loss_pos = -label*tf.log(prob+1e-6)
            # loss_neg = label*tf.log(1.-prob+1e-6)
            loss = loss_pos 
        loss = tf.reduce_sum(loss, 1, keep_dims=True)
        loss = nation_mean*loss
        # loss = nation_onehot*loss

 
        
        loss = tf.reduce_sum(loss, 0) 
        # loss = tf.reduce_mean(loss, 0) 
        cont_loss += 1.0
        
        loss_each = loss
        
        if i == 0:
            loss = loss_each
        
            nation_sp = ctx.get_config("nation")
            dic_nation2exp = {'TH':2,'VN':3,'SG':1,'MY':3,'ID':4,'PH':4}
            flag_exp = dic_nation2exp[nation_sp]

            flag_exp=3

            if flag_exp == 1 or flag_exp == 2:
                if flag_exp == 1:
                    pref_vecs = tf.eye(6)
                    pref_vecs = tf.concat([tf.random_uniform((1,6)),tf.ones((1,6))/6.],0)
                    pref_vecs = pref_vecs/(tf.reduce_sum(pref_vecs,1,keep_dims=True)+1e-6)
                if flag_exp == 2:
                    pref_vecs = tf.eye(6)
                    pref_vecs = tf.concat([tf.ones((1,6))/6.,tf.ones((1,6))/6.],0)
                    pref_vecs = pref_vecs/(tf.reduce_sum(pref_vecs,1,keep_dims=True)+1e-6)

                weights,size_list=get_variable_list('gate_feature_layer')
                grad_list = []
                for j in range(6):
                    grad = tf.gradients(loss[j], weights)
                    grad=grad_flatten(grad,size_list)
                    grad=tf.stop_gradient(grad)
                    grad_list.append(grad)
                    print('grad.get_shape()',grad.get_shape())
                grads = tf.stack(grad_list)
           
                loss_value = tf.stop_gradient(loss)

                i_task = 0

                # flag, weight_init=solver.get_d_paretomtl_init(grads,loss_value,pref_vecs,i_task)
                weight=solver.get_d_paretomtl(grads,loss_value,pref_vecs,i_task)

                # flag = tf.cast(flag,dtype=weight_init.dtype)
                # weight = flag * weight + (1.-flag)*weight_init

            if flag_exp == 3:
                weight = tf.random_uniform((1,6))
                weight = weight/(tf.reduce_sum(weight)+1e-6)
                
            if flag_exp == 4:
                weight = tf.ones((1,6))/6.

            print('flag_exp',flag_exp)
                
            
            weight = tf.stop_gradient(tf.reshape(weight,[-1]))

            loss_vec = loss
            
            loss =  tf.reduce_sum(loss * weight) 
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_param_regu = cont_loss * weight_decay * tf.add_n(reg_losses)
            # loss_sum += loss + loss_param_regu
            loss_sum += loss  
        elif i == 1:
            loss = loss_each
 
            loss_vec2 = loss
            loss =  tf.reduce_mean(loss  )
            loss_param_regu = cont_loss* weight_decay * tf.add_n(reg_losses)
            loss_sum += cate_align_weight * loss + loss_param_regu
            break
#         break

    # graph matching loss -------------------------------------
 
    loss_swd,show_list_swd = swd_loss(current_id_ays_list_swd,node_src_context_ays_swd,nation_exist,wd_idx,wd_idx_norm,
                       gate_current_id_deep_list_swd,gate_node_src_context_ays_swd,nation_weights=weight,gm_loss_weight=gm_loss_weight,
                       flag_dist=flag_dist,var_weight=var_weight,gate_weight=gate_weight,gate_th=gate_th,flag_gate_above=flag_gate_above,
                       same_nation_weight=same_nation_weight,flag_uniform_nation_weight=flag_uniform_nation_weight,
                       contain_same_cate=contain_same_cate)
 
    with tf.variable_scope("gate_pairwise_dnn", reuse=tf.AUTO_REUSE) as scope:
        loss_sgwd,show_list_sgwd = sgwd_loss(current_id_ays_list_sgwd,node_src_context_ays_sgwd,nation_exist,gwd_idx,gwd_idx_norm,
                            gate_current_id_deep_list_swd,gate_node_src_context_ays_swd,
                            ps_num, init_val, stddev,
                            mode='gwd',nation_weights=weight,gm_loss_weight=gm_loss_weight,flag_dist=flag_dist,var_weight=var_weight,
                            gate_weight=gate_weight,gate_th=gate_th,flag_gate_above=flag_gate_above,
                            same_nation_weight=same_nation_weight,flag_uniform_nation_weight=flag_uniform_nation_weight,
                            contain_same_cate=contain_same_cate)
     
        loss_scgwd,show_list_scgwd = sgwd_loss(current_id_ays_list_sgwd,node_id_ays_list_swd,nation_exist,cgwd_idx,cgwd_idx_norm,
                            gate_current_id_deep_list_swd,gate_node_id_ays_list_swd,
                            ps_num, init_val, stddev,
                            mode='cgwd',nation_weights=weight,gm_loss_weight=gm_loss_weight,flag_dist=flag_dist,var_weight=var_weight,
                            gate_weight=gate_weight,gate_th=gate_th,flag_gate_above=flag_gate_above,
                            same_nation_weight=same_nation_weight,flag_uniform_nation_weight=flag_uniform_nation_weight,
                            contain_same_cate=contain_same_cate)
    
 
    loss_regu = loss_swd + loss_sgwd + loss_scgwd

    #-----------------------------------------------------------
 
    global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.to_float(global_step)
    flag_global_step = greater_than_zero(global_step-500000.)

    loss_sum = loss_sum + 0.1 * loss_regu
         
    return loss_sum, [loss,loss_param_regu,loss_regu,loss_vec,weight,loss_vec2,
                     show_list_swd,
                     show_list_sgwd,
                     show_list_scgwd,flag_global_step]

def auc(input_list, num_thresholds=200, decay_rate=1):
    sim_list,nation_onehot,align_info  = input_list

    sim=sim_list[0]
    neg_num = tf.shape(sim)[1] - 1
    sample_num = tf.shape(sim)[0]
    labels_matrix = tf.concat([tf.ones([sample_num, 1], tf.int32), tf.zeros([sample_num, neg_num], tf.int32)], axis=1)
    labels = tf.reshape(labels_matrix, [-1, 1])

    predictions = tf.reshape(tf.nn.sigmoid(sim, name='sigmoid_auc'), [-1, 1])
    _, auc_op = tf.contrib.metrics.streaming_auc(predictions, labels, num_thresholds=num_thresholds,
                                                     decay_rate=decay_rate)
    return auc_op