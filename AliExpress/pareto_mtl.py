import numpy as np
import tensorflow as tf

def greater_than_zero(x):
    x=tf.maximum(0.0*x,x)
    x = tf.sign(x)
    return x

class ParetoMTL:
    def __init__(self):
        self.MAX_ITER = 20
        self.STOP_CRIT = 1e-5
    
    def _min_norm_element_from2(self,v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        gamma1,gamma2,gamma3 = 0.999,0.001,-1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2+1e-6) )
        cost1,cost2,cost3 = v1v1,v2v2,v2v2 + gamma3*(v1v2 - v2v2)
        cond1,cond2 = greater_than_zero(v1v2 -v1v1),greater_than_zero(v1v2-v2v2)
#         print(gamma1,gamma2,gamma3,cond1,cond2)
        gamma = cond1*gamma1 + (1.-cond1)*cond2*gamma2 + (1.-cond1)*(1.-cond2)*gamma3
        cost = cond1*cost1 + (1.-cond1)*cond2*cost2 + (1.-cond1)*(1.-cond2)*cost3
        return gamma, cost
    
    def _min_norm_2d(self,GG):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        
        a=tf.ones_like(GG)
        c=tf.matrix_band_part(a, -1, 0)
        d=1-c
        e=tf.where(d>0)
        ii = tf.concat([tf.expand_dims(e[:,0],1),tf.expand_dims(e[:,0],1)],1)
        ij = tf.concat([tf.expand_dims(e[:,0],1),tf.expand_dims(e[:,1],1)],1)
        jj = tf.concat([tf.expand_dims(e[:,1],1),tf.expand_dims(e[:,1],1)],1)
        
        xx = tf.gather_nd(GG,ii)
        xy = tf.gather_nd(GG,ij)
        yy = tf.gather_nd(GG,jj)
        
        c,d = self._min_norm_element_from2(xx, xy, yy)
        
        idx = tf.argmin(d)
        sol = [tf.to_float(tf.gather(ij,idx)),tf.gather(c,idx),tf.gather(d,idx)]

        return sol
    
    def _projection2simplex(self,y,m):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """ 
    #     m = tf.shape(y)[0]
        y = tf.reshape(y,[m,1])
        sorted_y = tf.contrib.framework.sort(y,axis=0)[-1::-1]
        tmax_f = tf.reduce_mean(y) - 1./tf.reduce_sum(tf.ones_like(y))
        tmpsum = tf.cumsum(sorted_y,axis=0)[:-1]  
        tmax = (tmpsum-1.)/tf.expand_dims(tf.cast(tf.to_float(tf.range(m-1)),dtype=tmpsum.dtype)+1.,1) 
        bool_greater = tf.greater(tmax , sorted_y[1:])
        flag_greater = tf.to_float(tf.greater(tf.reduce_sum(tf.to_float(bool_greater)),0))
        idx = tf.cast(tf.reduce_min(tf.where(bool_greater)[:,0]),dtype=tf.int32)
        idx = tf.minimum(tf.maximum(idx,0),m-2)
        tmax = tf.gather(tmax,idx)

        tmax_f = tf.cast(flag_greater,dtype=tmax.dtype)*tmax+(1.-tf.cast(flag_greater,dtype=tmax.dtype))*tmax_f


        return tf.maximum(0.*y,y - tmax_f)
    
    
    def _next_point(self,cur_val, grad, n):
        proj_grad = grad - tf.reduce_mean(grad)  
        
        idx_min = tf.argmin(grad[:,0])
        idx_max = tf.argmax(grad[:,0])
        
        idx_pos = tf.where(proj_grad>=0)[:,0]
        idx_neg = tf.where(proj_grad<=0)[:,0]
        
        idx_pos=tf.concat([idx_pos,[idx_max,idx_max]],0)
        idx_neg=tf.concat([idx_neg,[idx_min,idx_min]],0)
        
        tm1 = -1.0*tf.gather(cur_val,idx_neg)/(tf.gather(proj_grad,idx_neg)+1e-6)
        tm2 = (1.0 - tf.gather(cur_val,idx_pos))/(tf.gather(proj_grad,idx_pos)+1e-6)

        skippers = tf.reduce_sum(tf.to_float(tf.less(tm1,1e-7))) + tf.reduce_sum(tf.to_float(tf.less(tm2,1e-7)))

        tm1=tf.contrib.framework.sort(tm1,axis=0)
        tm2=tf.contrib.framework.sort(tm2,axis=0)
 
        t=tf.ones(1,dtype=tm1.dtype)
        bool_greater = tf.greater(tm1,1e-7)
        flag_greater = tf.cast(tf.to_float(tf.greater(tf.reduce_sum(tf.to_float(bool_greater)),0)),dtype=tm1.dtype)
        idx = tf.cast(tf.where(bool_greater)[:,0],dtype=tf.int32)
        idx = tf.concat([idx,[tf.shape(tm1)[0]-1]],0)
        t = tf.reduce_min(tf.gather(tm1,idx)) * flag_greater + t* (1.-flag_greater)

        bool_greater = tf.greater(tm2,1e-7)
        flag_greater = tf.cast(tf.to_float(tf.greater(tf.reduce_sum(tf.to_float(bool_greater)),0)),dtype=tm1.dtype)
        idx = tf.cast(tf.where(bool_greater)[:,0],dtype=tf.int32)
        idx = tf.concat([idx,[tf.shape(tm2)[0]-1]],0)
        t = tf.minimum(t, tf.reduce_min(tf.gather(tm2,idx))) * flag_greater + t* (1.-flag_greater)
 

        next_point = proj_grad*t + cur_val
        next_point = self._projection2simplex(next_point,n)
        return next_point
 

    def find_min_norm_element(self,GG):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
       
        init_sol = self._min_norm_2d(GG)
        
        n=tf.shape(GG)[0]
        idx= tf.to_float(tf.range(n))
        eq_i = tf.cast(tf.to_float(tf.equal(idx, init_sol[0][0])),dtype=init_sol[1].dtype)
        eq_j = tf.cast(tf.to_float(tf.equal(idx, init_sol[0][1])),dtype=init_sol[1].dtype)
        sol_vec = eq_i*init_sol[1]+eq_j*(1. - init_sol[1])
        sol_vec = tf.expand_dims(sol_vec,1)

        flag_simple = tf.to_float(tf.less(n, 3)) # This is optimal for n=2, so return the solution
        sol_vec_simple,cost_simple = sol_vec , init_sol[2]
     
        sol_vec_list = []
        nd_list = []
        change_list = []
        for _ in range(self.MAX_ITER):
            grad_dir = -1.0*tf.matmul(GG, sol_vec)
            new_point = self._next_point(sol_vec, grad_dir, n)
            
            v1v1 = tf.reduce_sum(tf.matmul(sol_vec, sol_vec, transpose_b=True)*GG)
            v1v2 = tf.reduce_sum(tf.matmul(sol_vec, new_point, transpose_b=True)*GG)
            v2v2 = tf.reduce_sum(tf.matmul(new_point, new_point, transpose_b=True)*GG)
             
            nc, nd = self._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            
            sol_vec_list.append(sol_vec)
            nd_list.append(nd)
            change_list.append(tf.reduce_sum(tf.abs(change)))
            
            sol_vec = new_sol_vec
        
        sol_vec_list=tf.stack(sol_vec_list)
        nd_list = tf.stack(nd_list)
        change_list = tf.stack(change_list)
        
        bool_less = tf.less(change_list , self.STOP_CRIT)
        flag_less = tf.to_float(tf.greater(tf.reduce_sum(tf.to_float(bool_less)),0))
        idx = tf.cast(tf.reduce_min(tf.where(bool_less)),dtype=tf.int32)
        idx = tf.minimum(tf.maximum(idx,0),tf.shape(change_list)[0]-1)
        
        sol_vec_save,nd_save = sol_vec,nd
        
        sol_vec = tf.gather(sol_vec_list,idx) * tf.cast(flag_less,dtype=sol_vec_list.dtype) + sol_vec * (1.-tf.cast(flag_less,dtype=sol_vec_list.dtype))
        nd = tf.gather(nd_list,idx) * tf.cast(flag_less,dtype=nd_list.dtype) + nd * (1.-tf.cast(flag_less,dtype=nd_list.dtype))
        
        sol_vec = sol_vec_simple * tf.cast(flag_simple,dtype=sol_vec_simple.dtype) + sol_vec * (1.-tf.cast(flag_simple,dtype=sol_vec.dtype))
        nd = cost_simple * tf.cast(flag_simple,dtype=nd_list.dtype) + nd * (1.-tf.cast(flag_simple,dtype=nd_list.dtype))
        
        
#         return sol_vec,grad_dir,new_point,v1v1, v1v2, v2v2,nc, nd,new_sol_vec,change
        return sol_vec,nd

    
    def get_d_paretomtl_init(self,grads,value,weights,i):
        """ 
        calculate the gradient direction for ParetoMTL initialization 
        """
 
        nobj = tf.shape(value)[0]
    
        weights=tf.cast(weights,dtype=value.dtype)

        #zero loss protection
        value_nonzero = greater_than_zero(value)
        value_nonzero = tf.reshape(value_nonzero,[nobj,])
        weights = weights * value_nonzero
        weights_eq_zero = 1.-greater_than_zero(tf.reduce_sum(weights,1,keep_dims=True))
        weights += weights_eq_zero * tf.ones_like(weights) * value_nonzero
        weights /= (tf.reduce_sum(weights,1,keep_dims=True)+1e-6)

        # check active constraints
        current_weight = weights[i]
        rest_weights = weights
        w =  current_weight - rest_weights
        
        value = tf.reshape(value,[nobj,1])
        value = tf.nn.l2_normalize(value,axis=0)

        gx =  tf.matmul(w,value)
        # idx = tf.greater(gx ,  -1e-5)
        idx = tf.greater(gx ,  0)
         
  
        flag_eq_0 = tf.cast(tf.equal(tf.reduce_sum(tf.to_float(idx)),0),dtype=w.dtype)
        flag_eq_1 = tf.cast(tf.equal(tf.reduce_sum(tf.to_float(idx)),1),dtype=w.dtype)
        flag_geq_2 = tf.cast(tf.greater(tf.reduce_sum(tf.to_float(idx)),1),dtype=w.dtype)
        
        idx_where = tf.cast(tf.where(idx)[:,0],dtype=tf.int32)
        
        idx_where = tf.concat([idx_where,[tf.shape(w)[0]-1,tf.shape(w)[0]-1]],0)
        
        w = tf.gather(w,idx_where)
        
        vec =  tf.matmul(w,grads)
        GG = tf.matmul(vec,vec,transpose_b=True)
        sol, nd = self.find_min_norm_element(GG)
          
         
        weight = flag_eq_0* tf.zeros(nobj,dtype=w.dtype) + \
                 flag_eq_1 * tf.reduce_sum(w,0) + \
                 flag_geq_2 * tf.reduce_sum(w*sol,0)
        
        flag = flag_eq_0
 

        return flag, weight

#         return w,idx_where,flag_eq_0,flag_eq_1,flag_geq_2,vec,GG


    def get_d_paretomtl(self,grads,value,weights,i):
        """ calculate the gradient direction for ParetoMTL """
        
        nobj = tf.shape(value)[0]
        
        weights=tf.cast(weights,dtype=value.dtype)

        #zero loss protection
        value_nonzero = greater_than_zero(value)
        value_nonzero = tf.reshape(value_nonzero,[nobj,])
        weights = weights * value_nonzero
        weights_eq_zero = 1.-greater_than_zero(tf.reduce_sum(weights,1,keep_dims=True))
        weights += weights_eq_zero * tf.ones_like(weights) * value_nonzero
        weights /= (tf.reduce_sum(weights,1,keep_dims=True)+1e-6)

        # check active constraints
        current_weight = weights[i]
        rest_weights = weights 
        w =  current_weight - rest_weights
         
        w = tf.concat([tf.cast(tf.eye(nobj),dtype=w.dtype),w],0)
        
        value = tf.reshape(value,[nobj,1])
        value = tf.nn.l2_normalize(value,axis=0)
  
        gx =  tf.matmul(w,value)
        # idx = tf.greater(gx ,  -1e-5)
        idx = tf.greater(gx ,  0)
        idx_where = tf.cast(tf.where(idx)[:,0],dtype=tf.int32)
        
        w = tf.gather(w,idx_where)
        
        vec =  tf.matmul(w,grads)
        GG = tf.matmul(vec,vec,transpose_b=True)
        sol, nd = self.find_min_norm_element(GG)
        
        weight = tf.reduce_sum(w*sol,0)
 
        return weight
