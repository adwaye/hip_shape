"""Contains some methods used to align hips"""
import numpy as np
from scipy import ndimage as ndi
from jax.scipy import ndimage as jndi
import jax.scipy.optimize as optimize
import jax.numpy as jnp
import edt #got this from https://github.com/seung-lab/euclidean-distance-transform-3d
import time
dtype = jnp.float32

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




def _make_distance_transform(points,size,ncpus=1):
    """Generates a distance array of shape size where len(size)=3 where the distance array is calculated from the points takes in a set of points where np.max(points,axis=0) < size

    :param points: [N,3] np array of points representing the set of points from which the distance transform is being calculated
                   points need to lie within [0,size[0]]\times[0,size[1]]\times[0,size[2]]
    :param size:   tuple, array of length 3
    :return: array of sshape size whose values represent the minimum distance from the set of points
    """
    #making a mask to be used by edt.edt
    mask = np.zeros(size.ravel())
    #"""
    for pt in points.astype(int):
        #print(pt)
        mask[pt[0],pt[1],pt[2]] = 1

    mask      = mask.astype(bool)

    ncpus=4
    dt = edt.edt(data=~mask,anisotropy=(1,1,1), black_border=False,order='C',parallel=ncpus)
    return dt


def _embed_points_in_array(points1,points2,padding=(100,100)):
    """Translates points1 and points2 so that they live in a square domain [0,size[0]]\times[0,size[1]]\times[0,size[2]]

    :param points1: np.array [N1,3]
    :param points2: np.array [N2,3]
    :param padding: padding to be applied so that size-np.max([points1,points2],axis=0)>padding[0] and np.max([points1,
                    points2],axis=0)-0>padding[1]
                    padding to be added on either side of embedding array
    :return: points1_proc np.array [N1,3] points1 translated
            ,points2_proc np.array [N1,3] points2 translated
            ,size tuple shape of array that would contain the above 2 points
    """
    assert points1.shape[-1]==points2.shape[-1]
    assert len(padding) == 2
    padding_plus  = padding[0]
    padding_minus = padding[1]
    points1_cent  = points1-np.mean(points1,axis=0,keepdims=True)
    points2_cent  = points2-np.mean(points2,axis=0,keepdims=True)

    merged_pts = np.concatenate([points1_cent,points2_cent],axis=0)
    max_array  = np.max(merged_pts,axis=0,keepdims=True)+padding_plus #add some padding
    min_array  = np.min(merged_pts,axis=0,keepdims=True)-padding_minus
    size       = max_array-min_array
    size       = size.astype(int)

    points1_proc = points1_cent+size//2
    points2_proc   = points2_cent+size//2

    return points1_proc,points2_proc,size


def eval_distance(points,dist):
    """finds the value of the scalar field dist with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,dist.shape[1]]

    :param points: np.array [N,3] list of points wwhere the scalar field given by dist needs to be evaluated
    :param dist:   scalar array with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,
                   dist.shape[1]] np.array size [Nx,Ny,Nz]
    :return: array of values [N]
    """
    assert len(points.shape)==2
    if points.shape[1]!=3:
        points=points.transpose()
    xn = points[:,1]#.astype(np.int)
    yn = points[:,0]#.astype(np.int)
    zn = points[:,2]#.astype(np.int)

    dist_vals = ndi.interpolation.map_coordinates(dist, [yn, xn, zn], order=2)
    return dist_vals


def jax_eval_distance(points,dist):
    """finds the value of the scalar field dist with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,dist.shape[1]]

    :param points: np.array [N,3] list of points wwhere the scalar field given by dist needs to be evaluated
    :param dist:   scalar array with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,
                   dist.shape[1]] np.array size [Nx,Ny,Nz]
    :return: array of values [N]
    """
    assert len(points.shape) == 2
    if points.shape[1] != 3:
        points = points.transpose()
    xn = points[:,1]#.astype(np.int)
    yn = points[:,0]#.astype(np.int)
    zn = points[:,2]#.astype(np.int)
    #print(f'z shape is {zn.shape}')
    dist_vals = jndi.map_coordinates(dist, [yn, xn, zn], order=1)
    return dist_vals


def jax_yaw_matrix(angle=np.pi/2):
    """Returns the yaw matrix with angle (rotation around z-axis) see https://en.wikipedia.org/wiki /Rotation_matrix in_three_dimensions

    :param angle:
    :return:
    #https: // en.wikipedia.org / wiki / Rotation_matrix  #In_three_dimensions
    """
    out = jnp.array([[jnp.cos(angle) , -jnp.sin(angle), 0],
                     [jnp.sin(angle) ,  jnp.cos(angle), 0],
                     [0               , 0,               1]
                     ],dtype=dtype)
    return out


def jax_pitch_matrix(angle):
    """Returns the pitch matrix with angle (rotation around y-axis) see https://en.wikipedia.org/wiki /Rotation_matrix in_three_dimensions

    :param angle:
    :return:
    #https: // en.wikipedia.org / wiki / Rotation_matrix  #In_three_dimensions
    """
    out = jnp.array([[jnp.cos(angle) , 0 , jnp.sin(angle)],
                     [0              , 1 , 0             ],
                     [-jnp.sin(angle), 0 , jnp.cos(angle)]
                     ],dtype=dtype)
    return out

def jax_roll_matrix(angle):
    """Returns the roll matrix with angle (rotation around x-axis) see https://en.wikipedia.org/wiki /Rotation_matrix in_three_dimensions

    :param angle:
    :return:

    """
    out = jnp.array([[1,             0 ,               0],
                     [0,jnp.cos(angle) , -jnp.sin(angle)],
                     [0,jnp.sin(angle) ,  jnp.cos(angle)]
                     ],dtype=dtype)
    return out


def jax_rotation_matrix3d(yaw_angle,pitch_angle,roll_angle):
    """Build a 3d rotation matrix see https://en.wikipedia.org/wiki /Rotation_matrix in_three_dimensions

    :param yaw_angle: z-axis rotation angle:param yaw_angle:
    :type yaw_angle: float scalar
    :param pitch_angle: y-axis rotation angle
    :type pitch_angle: float scalar
    :param roll_angle: z-axis rotation angle
    :type roll_angle: float scalar
    :return: 3d rotation matrix
    :rtype: jax array shape (3,3)
    """

    yaw_mat   = jax_yaw_matrix(yaw_angle)
    pitch_mat = jax_pitch_matrix(pitch_angle)
    roll_mat  = jax_roll_matrix(roll_angle)

    out = jnp.matmul(pitch_mat,roll_mat)
    out = jnp.matmul(yaw_mat,out)
    return out



def affine_cost(affine_params,points,dist):
    """

    :param affine_params: affine parameters
                          yaw_angle   = affine_params[0]
                          pitch_angle = affine_params[1]
                          roll_angle  = affine_params[2]
                          translation = affine_params[3:]
    :type affine_params: array
    :param points:
    :type points:
    :param dist:
    :type dist:
    :return:
    :rtype:
    """
    assert affine_params.shape == (6,)
    yaw_angle   = affine_params[0]
    pitch_angle = affine_params[1]
    roll_angle  = affine_params[2]
    translation = jnp.expand_dims(affine_params[3:],axis=1)
    if points.shape[0]!=3:
        points = points.transpose()


    return _affine_cost(yaw_angle,pitch_angle,roll_angle,translation,points,dist)

def _affine_cost(yaw_angle,pitch_angle,roll_angle,translation,points,dist):
    """returns the sum of the distance transform evaluated at the affine transformed points with rotation matrix
    given by yaw_angle,pitch_angle,roll_angle and translation given by translation

    :param yaw_angle: scalar  float
    :param pitch_angle: scalar float
    :param roll_angle:  scalar float
    :param translation:  array shape [3]
    :param dist: distance transform on which to evaluate scalar
    :return:
    """
    assert points.shape[0]==3
    assert translation.shape == (3,1)
    rot_matrix  = jax_rotation_matrix3d(yaw_angle,pitch_angle,roll_angle)

    mean_points  =  jnp.mean(points,axis=1,keepdims=True)
    trans_points =  jnp.add(jnp.matmul(rot_matrix,points-mean_points),translation)
    #print(f'transformed points have shape {trans_points.shape}')
    dist_vals    = jax_eval_distance(trans_points,dist)
    return jnp.divide(jnp.sum(dist_vals),len(points))



def align_Planes_bfgs(plane1, plane2):
    """rotates two planes so that they are parallel to each other. Uses bfgs to find optimal yaw, pitch and roll angle that minimizes the distance between the normal of plane1 and the rotated normal of plane2


    :param plane1: reference plane
    :type plane1: skspatial.objects.plane.Plane
    :param plane2: skspatial.objects.plane.Plane
    :type plane2: plane to be rotated
    :return: rotation matrix mapping plane2 onto plane1
    :rtype: jnp.array
    """
    normal1 = jnp.array(plane1.normal,dtype=dtype)
    normal2 = jnp.array(plane2.normal,dtype=dtype)

    params = np.array([0.0,0.0,0.0])

    def cost_function(params,normal1,normal2):
        rot_matrix = jax_rotation_matrix3d(params[0],params[1],params[2])
        cost = jnp.sum(jnp.square(normal1-jnp.matmul(rot_matrix,normal2)))
        return cost


    from functools import partial
    # loss_fn = partial(cost_function,normal1=normal1,normal2=normal2)
    loss_fn = lambda z: cost_function(z, normal1, normal2)


    res = optimize.minimize(loss_fn,params,method='BFGS',options={'maxiter':100})

    rot_matrix = jax_rotation_matrix3d(*res.x)

    return rot_matrix
