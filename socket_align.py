
import os.path



from read_landmarks import *
from datetime import datetime
import pickle

import numpy as np
import numpy.linalg
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
from utils import FourierFitter
from scipy import ndimage as ndi
from jax.scipy import ndimage as jndi
import jax.scipy.optimize as optimize
import jax.numpy as jnp
import edt #got this from https://github.com/seung-lab/euclidean-distance-transform-3d
import time

source_loc = './data/Segmentation_and_landmarks_raw/'
target_loc = './data/Segmentation_and_landmarks_processed/'
APP_aligned_loc = './data/Segmentation_and_landmarks_APP_aligned/'
Socket_aligned_trans = './data/Segmentation_and_landmarks_socket_aligned/'

def _extract_data():
    dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")



    paths         = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']
    location      = os.path.join(target_loc,paths[0])
    files         = [os.path.join(location,f) for f in os.listdir(location)]
    template_file = files[5]
    target_file   = files[10]

    # location = os.path.join(target_loc,paths[0])
    # files = [os.path.join(location,f) for f in os.listdir(location)]


    with open(template_file,'rb') as fp:
        template_data = pickle.load(fp)

    with open(target_file,'rb') as fp:
        target_data = pickle.load(fp)


    for side in ['Right']:#['Right','Left']:
        k = 0
        for key in ['Ant Lat','Post Lat']:
            try:
                if k == 0:
                    template_points = template_data['landmarks'][side+' '+key]
                    print(template_points.shape)
                    target_points   = target_data['landmarks'][side+' '+key]
                else:
                    template_points_ = template_data['landmarks'][side+' '+key]
                    print(template_points_.shape)
                    template_points  = np.concatenate((template_points,template_points_),axis=0)
                    target_points_   = target_data['landmarks'][side+' '+key]
                    target_points    = np.concatenate((target_points,target_points_),axis=0)
                k+=1
            except KeyError:
                print('cannot find '+side+' '+key)
    return template_points,target_points
# plane = Plane.best_fit(Points(template_points))
#
# fig = plt.figure(figsize=(4,4))
#
# ax = fig.add_subplot(111,projection='3d')
#
# ax.set_box_aspect(aspect=(1,1,1))
# ax.plot(template_points[:,0],template_points[:,1],template_points[:,2],color='blue')
# lims = np.min(template_points,axis=0),np.max(template_points,axis=0)
#plane.plot_3d(ax,lims_y=(-150,50),lims_x=(-75,50),alpha=0.2)








def _make_distance_transform(points,size,ncpus=1):
    """
    Generates a distance array of shape size where len(size)=3 where the distance array is calculated from the points
    takes in a set of points where np.max(points,axis=0) < size
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
    """
    Translates points1 and points2 so that they live in a square domain [0,size[0]]\times[0,size[1]]\times[0,size[2]]

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
    """
    finds the value of the scalar field dist with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,
    dist.shape[1]]
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
    """
    finds the value of the scalar field dist with support [0,dist.shape[0]]\times[0,dist.shape[1]]\times[0,
    dist.shape[1]]
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
    """
    Returns the yaw matrix with angle (rotation around z-axis)
    :param angle:
    :return:
    #https: // en.wikipedia.org / wiki / Rotation_matrix  #In_three_dimensions
    """
    out = jnp.array([[jnp.cos(angle) , -jnp.sin(angle), 0],
                     [jnp.sin(angle) ,  jnp.cos(angle), 0],
                     [0               , 0,               1]
                     ])
    return out


def jax_pitch_matrix(angle):
    """
    Returns the pitch matrix with angle (rotation around y-axis)
    :param angle:
    :return:
    #https: // en.wikipedia.org / wiki / Rotation_matrix  #In_three_dimensions
    """
    out = jnp.array([[jnp.cos(angle) , 0 , jnp.sin(angle)],
                     [0              , 1 , 0             ],
                     [-jnp.sin(angle), 0 , jnp.cos(angle)]
                     ])
    return out

def jax_roll_matrix(angle):
    """
    Returns the roll matrix with angle (rotation around x-axis)

    :param angle:
    :return:
    #https: // en.wikipedia.org / wiki / Rotation_matrix  #In_three_dimensions
    """
    out = jnp.array([[1,             0 ,               0],
                     [0,jnp.cos(angle) , -jnp.sin(angle)],
                     [0,jnp.sin(angle) ,  jnp.cos(angle)]
                     ])
    return out


def jax_rotation_matrix3d(yaw_angle,pitch_angle,roll_angle):
    """Build a 3d rotation matrix

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
    """
    returns the sum of the distance transform evaluated at the affine transformed points with rotation amtrix given by
    yaw_angle,pitch_angle,roll_angle and translation given by translation
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



def align_Planes(plane1, plane2):
    """rotates two planes so that they are parallel to each other. Uses bfgs to find optimal yaw,
    pitch and roll angle that minimizes the distance between the normal of plane1 and the rotated normal of plane2


    :param plane1: reference plane
    :type plane1: skspatial.objects.plane.Plane
    :param plane2: skspatial.objects.plane.Plane
    :type plane2: plane to be rotated
    :return: rotation matrix mapping plane2 onto plane1
    :rtype: jnp.array
    """
    normal1 = jnp.array(plane1.normal)
    normal2 = jnp.array(plane2.normal)

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






def _test_modules():
    template_points,target_points = _extract_data()
    assert template_points.shape[-1]==target_points.shape[-1]
    #assert not np.equal(template_points,target_points)

    template_points_proc,target_points_proc,size = _embed_points_in_array(template_points,target_points)
    print(f'size of embedded template points is {template_points_proc.shape}')
    #check dimension
    assert len(size.ravel())==3
    #check embedding
    assert np.all((size - np.max(template_points_proc,keepdims=True,axis=0))>0)
    assert np.all((size - np.max(target_points_proc,keepdims=True,axis=0)) > 0)
    assert np.all(target_points_proc > 0)
    assert np.all(template_points_proc > 0)
    dt = _make_distance_transform(points = template_points_proc,size=size,ncpus=4)
    #check dimension of dt matches dimension of size
    assert np.all(dt.shape==size)
    dist_vals = eval_distance(template_points_proc,dist=dt)
    dist      = np.sum(dist_vals)/template_points_proc.shape[0]
    print(f'distance for templare_points_proc is {dist}')
    #assert dist < 10e-3
    dist_vals_exact = eval_distance(template_points_proc.astype(np.int),dist=dt)
    dist_exact      = np.sum(dist_vals)/template_points_proc.shape[0]
    assert dist_exact<10-3

    dist_vals_exact = jax_eval_distance(template_points_proc.astype(np.int),dist=dt)
    dist_exact      = np.sum(dist_vals)/template_points_proc.shape[0]
    assert dist_exact<10-3

    for ncpus in  [1,2,4]:
        start = time.time()

        dt = _make_distance_transform(points = template_points_proc,size=size,ncpus=ncpus)
        time_taken_edt = time.time()-start
        my_string = "time for github dt {:} with {:} cpus".format(time_taken_edt,ncpus)
        print(my_string)


    points = jnp.array([1,1,1])
    angle  = jnp.pi/2
    Rx = jax_roll_matrix(angle)
    Ry = jax_pitch_matrix(angle)
    Rz = jax_yaw_matrix(angle)
    rot_matrix = jax_rotation_matrix3d(angle,angle,angle)
    assert jnp.linalg.norm(jnp.matmul(Rx,points) - jnp.array([1,-1,1]))<10e-5
    assert jnp.linalg.norm(jnp.matmul(Ry,points) - jnp.array([1,1,-1]))<10e-5
    assert jnp.linalg.norm(jnp.matmul(Rz,points) - jnp.array([-1,1,1]))<10e-5
    assert jnp.linalg.norm(rot_matrix-jnp.matmul(Rz,jnp.matmul(Ry,Rx)))<10e-5
    rot_matrix = jax_rotation_matrix3d(0.0,0.0,0.0)

    print(f'Rotation matrix size {rot_matrix.shape}')
    print(f'points  size {template_points_proc.shape}')
    print(f'points transpose size {template_points_proc.transpose().shape}')

    assert jnp.linalg.norm(jnp.matmul(rot_matrix,template_points_proc.transpose())-template_points_proc.transpose())<10e-5

    print(jnp.matmul(rot_matrix,points))

    params = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
    print('evaluating distance')
    loss = affine_cost(params,template_points_proc,dt)
    assert loss<10-3










#need to embed the curves into an array:
#calculating size of the array to allocate for the distance transform
if __name__=='__main__':
    # _test_modules()

    from jax_transformations3d import jax_transformations3d as jts


    template_points,target_points = _extract_data()
    template_points_mean  = np.mean(template_points,axis=0,keepdims=True)
    target_points_mean = np.mean(template_points,axis=0,keepdims=True)
    template_points -= template_points_mean
    target_points -= target_points_mean

    template_plane = Plane.best_fit(Points(template_points))
    target_plane = Plane.best_fit(Points(target_points))

    normal1 = jnp.array(template_plane.normal)
    normal2 = jnp.array(target_plane.normal)

    # params = np.array([0.0,0.0,0.0])
    #
    # def cost_function(params,normal1,normal2):
    #     rot_matrix = jax_rotation_matrix3d(params[0],params[1],params[2])
    #     cost = jnp.sum(jnp.square(normal1-jnp.matmul(rot_matrix,normal2)))
    #     return cost
    #
    #
    # from functools import partial
    # # loss_fn = partial(cost_function,normal1=normal1,normal2=normal2)
    # loss_fn = lambda z: cost_function(z, normal1, normal2)
    #
    #
    # res = optimize.minimize(loss_fn,params,method='BFGS',options={'maxiter':100})
    #
    # rot_matrix = jax_rotation_matrix3d(*res.x)
    rot_matrix = align_Planes(template_plane, target_plane)
    normal2_mapped = jnp.matmul(rot_matrix,normal2)

    print(f'maped nnormal {normal2_mapped}')
    print(f'original nnormal {normal1}')






























    # template_points_proc,target_points_proc,size = _embed_points_in_array(template_points,target_points,padding=(300,
    #                                                                                                              300))
    # dt = _make_distance_transform(points=template_points_proc,size=size,ncpus=2)
    #
    #
    # from functools import partial
    # method = 'BFGS'
    # nits =100
    # loss_fun = partial(affine_cost,points=target_points_proc,dist=dt)
    # fig = plt.figure(figsize=(4,4))
    #
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.set_box_aspect(aspect=(1,1,1))
    # # ax.plot(template_points_proc[:,0],template_points_proc[:,1],template_points_proc[:,2],color='blue',label='Socket plane points')
    # ax.scatter(template_points_proc[:,0],template_points_proc[:,1],template_points_proc[:,2],color='blue',
    #         label='Socket plane points')
    # point_plot = ax.scatter(target_points_proc[:,0],target_points_proc[:,1],target_points_proc[:,2],color='red',
    #                      label='Transformed points')
    # plt.show()
    # def callback_function(new_params):
    #     rot_matrix = jax_rotation_matrix3d(new_params[0],new_params[1],new_params[3])
    #     trans_points = jnp.add(jnp.matmul(rot_matrix,target_points_proc),translation)
    #     point_plot.set_offsets(np.c_[trans_points[:,0],trans_points[:,1],trans_points[:,2]])
    #     plt.title(f'{callback_function.nits} iterations')
    #
    #
    # #options = {'disp':False}
    # options = {}
    # options['maxiter'] = nits
    # params = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
    # #
    # res = optimize.minimize(loss_fun,params,method=method,options=options)#,callback=callback_function)
    # new_params = res.x
    # rot_matrix = jax_rotation_matrix3d(new_params[0],new_params[1],new_params[2])
    # trans_points = jnp.add(jnp.matmul(rot_matrix,target_points_proc.transpose()),jnp.expand_dims(new_params[3:],axis=1))
    # rot_points   = jnp.matmul(rot_matrix,target_points_proc.transpose())
    # point_plot = ax.scatter(trans_points[0,:],trans_points[1,:],trans_points[2,:],color='black',
    #                      label='Transformed points')




    #test_modules()









