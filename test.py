
import os.path

import matplotlib.pyplot as plt

from read_landmarks import *
from datetime import datetime
import pickle
from mayavi import mlab
import numpy as np
import numpy.linalg
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
from mpl_toolkits.mplot3d import Axes3D
from utils import FourierFitter
from scipy import ndimage as ndi
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
    target_file   = files[2]

    # location = os.path.join(target_loc,paths[0])
    # files = [os.path.join(location,f) for f in os.listdir(location)]


    with open(template_file,'rb') as fp:
        template_data = pickle.load(fp)

    with open(target_file,'rb') as fp:
        target_data = pickle.load(fp)


    for side in ['Right']:#['Right','Left']:
        k = 0
        for key in ['Ant Lat']:#,'Post Lat']:
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
    for pt in points.astype(np.int):
        #print(pt)
        mask[pt[0],pt[1],pt[2]] = 1

    mask      = mask.astype(np.bool)

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
    size       = size.astype(np.int)

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
    xn = points[:,1]#.astype(np.int)
    yn = points[:,0]#.astype(np.int)
    zn = points[:,2]#.astype(np.int)
    dist_vals = ndi.interpolation.map_coordinates(dist, [yn, xn, zn], order=1)
    return dist_vals


def test_modules():
    template_points,target_points = _extract_data()
    assert template_points.shape[-1]==target_points.shape[-1]
    #assert not np.equal(template_points,target_points)

    template_points_proc,target_points_proc,size = _embed_points_in_array(template_points,target_points)
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
    dist_vals = eval_distance(template_points,dist=dt)
    dist      = np.sum(dist_vals)/template_points.shape[0]
    assert dist < 10e-3
    dist_vals_exact = eval_distance(template_points.astype(np.int),dist=dt)
    dist_exact      = np.sum(dist_vals)/template_points.shape[0]
    assert dist_exact==0

    for ncpus in  [1,2,4]:
        start = time.time()

        dt = _make_distance_transform(points = template_points_proc,size=size,ncpus=ncpus)
        time_taken_edt = time.time()-start
        my_string = "time for github dt {:} with {:} cpus".format(time_taken_edt,ncpus)
        print(my_string)









#need to embed the curves into an array:
#calculating size of the array to allocate for the distance transform
if __name__=='__main__':

    test_modules()






