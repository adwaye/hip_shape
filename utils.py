import numpy as np


class FourierFitter(object):
    def __init__(self,x,y,order=20):
        self._x = x
        self._y = y
        self._order = order
        self.get_locust()
        self.get_coeffs()



    def get_locust(self):
        """
        returns the A0, C0 from
        Nonlinear Shape Manifolds as Shape Priors in Level Set Segmentation and
        Tracking
        :return:
        """
        x = self._x
        y = self._y

        K = len(x)
        delta_x = np.diff(x)
        delta_y = np.diff(y)
        delta_t = np.sqrt(delta_x ** 2 + delta_y ** 2)
        t_array = np.concatenate([([0.0]),np.cumsum(delta_t)])

        perimeter = np.sum(delta_t)
        psi_x = np.cumsum(delta_x) - (delta_x / delta_t) * t_array[1:]
        psi_y = np.cumsum(delta_y) - (delta_y / delta_t) * t_array[1:]

        t_diff = np.diff(t_array)
        t_diff[0] = 0
        t_diff2 = np.diff(t_array ** 2)
        t_diff2[0] = 0
        A0 = (1 / perimeter) * np.sum((delta_x / (2 * delta_t)) * t_diff2 + psi_x * t_diff) + x[0]
        C0 = (1 / perimeter) * np.sum((delta_x / (2 * delta_t)) * t_diff2 + psi_x * t_diff) + x[0]
        self.A0 = A0
        self.C0 = C0


    def get_coeffs(self):
        x = self._x
        y = self._y

        delta_x = np.diff(x)
        delta_y = np.diff(y)
        delta_t = np.sqrt(delta_x ** 2 + delta_y ** 2)
        t_array = np.concatenate([([0.0]),np.cumsum(delta_t)])

        perimeter = np.sum(delta_t)
        psi_x = np.cumsum(delta_x) - (delta_x / delta_t) * t_array[1:]
        psi_y = np.cumsum(delta_y) - (delta_y / delta_t) * t_array[1:]

        t_diff = np.diff(t_array)
        t_diff[0] = 0
        t_diff2 = np.diff(t_array ** 2)
        t_diff2[0] = 0
        A0 = (1 / perimeter) * np.sum((delta_x / (2 * delta_t)) * t_diff2 + psi_x * t_diff) + x[0]
        C0 = (1 / perimeter) * np.sum((delta_y / (2 * delta_t)) * t_diff2 + psi_y * t_diff) + y[0]
        order = self._order
        k = np.arange(1,order + 1)
        constants = perimeter / (k * k * 2 * np.pi * np.pi)
        k = np.expand_dims(k,axis=0)
        t_array_e = np.expand_dims(t_array,axis=1)
        arg_array = k * t_array_e * 2 * np.pi / perimeter

        cos_array = np.cos(arg_array)
        sin_array = np.sin(arg_array)

        a_k = np.concatenate(  ([A0],
                              constants*np.sum( np.expand_dims(delta_x/delta_t,axis=1)*np.diff(cos_array,axis=0)
                                                ,axis=0))  )
        b_k = np.concatenate(  ([0.0],
                              constants*np.sum( np.expand_dims(delta_x/delta_t,axis=1)*np.diff(sin_array,axis=0)
                               ,axis=0)))

        c_k = np.concatenate(([C0],constants*np.sum( np.expand_dims(delta_y/delta_t,axis=1)*np.diff(cos_array,axis=0)
                                                     ,
                                               axis=0)))
        d_k = np.concatenate(([0.0],constants*np.sum( np.expand_dims(delta_y/delta_t,axis=1)*np.diff(sin_array,axis=0)
                                                     ,
                                               axis=0)))

        self.coeffs =  np.concatenate(
        [
            a_k.reshape((order+1, 1)),
            b_k.reshape((order+1, 1)),
            c_k.reshape((order+1, 1)),
            d_k.reshape((order+1, 1)),
        ],
        axis=1,
    )

    def sample_pts(self,t):
        a = self.coeffs[:,0]
        b = self.coeffs[:,1]
        c = self.coeffs[:,2]
        d = self.coeffs[:,3]
        return sample_fourier(t,a,b,c,d)



def sample_fourier(t,a,b,c,d):
    if len(t.shape)==2:
        t = np.expand_dims(t,axis=-1)
    B = t.shape[0]
    K = a.shape[0]

    k_seq = np.reshape(2*np.pi*np.arange(0,K,1),(1,1,K))
    k_t_arr = np.multiply(t,k_seq)

    cos_arr = np.cos(k_t_arr)
    sin_arr = np.sin(k_t_arr)

    x_terms = np.multiply(cos_arr,a) + np.multiply(sin_arr,b)
    y_terms = np.multiply(cos_arr,c) + np.multiply(sin_arr,d)

    tf_x = np.sum(x_terms,axis=2,keepdims=True)
    tf_y = np.sum(y_terms,axis=2,keepdims=True)

    sampled_pts = np.concatenate((tf_x,tf_y),axis=-1)


    return sampled_pts


# def tf_fourier_curve_equation(tf_t,tf_a,tf_b,tf_c,tf_d):
#     """
#     x[i] = \sum_{k=0}^K-1 a[k]*cos(2*pi*k*t[i]) + b[k]*sin(2*pi*k*t[i])
#     y[i] = \sum_{k=0}^K-1 c[k]*cos(2*pi*k*t[i]) + d[k]*sin(2*pi*k*t[i])
#     :param tf_t: shape [B,N,1]
#     :param tf_a: shape [B,1,K]
#     :param tf_b: shape [B,1,K]
#     :param tf_c: shape [B,1,K]
#     :param tf_d: shape [B,1,K]
#     :return:
#     """
#
#     if tf_t.shape._v2_behavior:
#         B = tf_t.shape[0]
#         K = tf_a.shape[2]
#     else:
#         B = tf_t.get_shape()[0]
#         K = tf_a.get_shape()[2]
#
#     tf_k_seq   = np.reshape(2*np.pi*np.arange(0,K,1) ,(1,1,K)  ) #[0,2pi,6pi,8pi,....]
#     tf_k_t_arr = tf.multiply(tf_t,tf_k_seq)  #[]
#
#     tf_cos_arr = tf.cos(tf_k_t_arr)
#     tf_sin_arr = tf.sin(tf_k_t_arr)
#
#     tf_x_terms = tf.multiply(tf_cos_arr,tf_a)+tf.multiply(tf_sin_arr,tf_b)
#     tf_y_terms = tf.multiply(tf_cos_arr,tf_c)+tf.multiply(tf_sin_arr,tf_d)
#
#     tf_x = tf.reduce_sum(tf_x_terms,axis=2,keepdims=True)
#     tf_y = tf.reduce_sum(tf_y_terms,axis=2,keepdims=True)
#
#     tf_sampled_pts = tf.concat([tf_x,tf_y],axis=-1)
#
#
#
#     return tf_sampled_pts