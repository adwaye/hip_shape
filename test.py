
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




dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")

source_loc           = './data/Segmentation_and_landmarks_raw/'
target_loc           = './data/Segmentation_and_landmarks_processed/'
APP_aligned_loc      = './data/Segmentation_and_landmarks_APP_aligned/'
Socket_aligned_trans = './data/Segmentation_and_landmarks_socket_aligned/'

paths     = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']

location = os.path.join(target_loc,paths[0])
files = [os.path.join(location,f) for f in os.listdir(location)]
template_file = files[0]
target_file = files[2]

location = os.path.join(target_loc,paths[0])
files = [os.path.join(location,f) for f in os.listdir(location)]
template_file = files[5]
target_file = files[2]

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


#need to embed the curves into an array:
#calculating size of the array to allocate for the distance transform
padding_plus  = 100
padding_minus = 100

template_points_cent = template_points-np.mean(template_points,axis=0,keepdims=True)
target_points_cent   = target_points-np.mean(target_points,axis=0,keepdims=True)

merged_pts = np.concatenate([template_points_cent,target_points_cent],axis=0)
max_array  = np.max(merged_pts,axis=0,keepdims=True)+padding_plus #add some padding
min_array  = np.min(merged_pts,axis=0,keepdims=True)-padding_minus
size       = max_array-min_array
size       = size.astype(np.int)

template_points_proc = template_points_cent+size//2
target_points_proc   = target_points_cent+size//2


mask = np.zeros(size.ravel())
"""
for points in template_points_proc.astype(np.int):
    print(points)
    mask[points[0],points[1],points[2]] = 1
"""
mask[100,:,:] = 1#todo try on different slices to see if this is working and plot the slices to see how the distance
# transform works
mask      = mask.astype(np.bool)

ncpus=4
dt = edt.edt(data=~mask,anisotropy=(1,1,1), black_border=False,order='C',parallel=ncpus)

slice = 100
plt.figure()
plt.title(label='mask')
plt.imshow(mask[slice,:,:
           ])
plt.colorbar()
plt.figure()
plt.title(label='distance transform')
plt.imshow(dt[slice,:,:
           ])
plt.colorbar()




def test_edt():
    log_loc = './log'
    log_file = os.path.join(log_loc,'edt_timing.txt')
    print("distance transform github")
    ncpus = 1
    start = time.time()
    dt = edt.edt(data=mask,anisotropy=(1,1,1), black_border=True,order='F',parallel=ncpus)
    time_taken_edt = time.time()-start
    my_string = "time for github dt {:} with {:} cpus".format(time_taken_edt,ncpus)
    with open(log_file,'w') as f:
        f.write( my_string+ ' \n')
    print(my_string)


    print("distance transform github")
    start = time.time()
    ncpus = 2
    dt = edt.edt(data=mask,anisotropy=(1,1,1), black_border=True,order='F',parallel=ncpus)
    time_taken_edt = time.time()-start
    my_string = "time for github dt {:} with {:} cpus".format(time_taken_edt,ncpus)
    with open(log_file,'a') as f:
        f.write( my_string+ ' \n')
    print(my_string)


    print("distance transform github")
    start = time.time()
    ncpus = 4
    dt = edt.edt(data=mask,anisotropy=(1,1,1), black_border=True,order='F',parallel=ncpus)
    time_taken_edt = time.time()-start
    my_string = "time for github dt {:} with {:} cpus".format(time_taken_edt,ncpus)
    with open(log_file,'a') as f:
        f.write( my_string+ ' \n')
    print(my_string)



# import plotly.graph_objects as go
# X = np.arange(0,size[0][0],1)
# Y = np.arange(0,size[0][1],1)
# Z = np.arange(0,size[0][2],1)
# fig = go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=dt.flatten(),
#     isomin=-0.0,
#     isomax=10,
#     opacity=0.1, # needs to be small to see through all surfaces
#     surface_count=21, # needs to be a large number for good volume rendering
#     ))
# fig.show()


# print("distance transform scipy")
# start = time.time()
# distArray = ndi.distance_transform_edt(~mask)
# time_taken_scipy = time.time()-start
# print("time for github scipy {:}".format(time_taken_scipy))






#
#
# Nmax = 2
# template_curve[:,0:int(D / 2)] = template_curve[:,0:int(D / 2)] + nNx / 2
# template_curve[:,int(D / 2):D] = template_curve[:,int(D / 2):D] + nNy / 2
# pts = np.transpose(
#     np.concatenate((template_curve[:,0:int(D / 2)],template_curve[:,int(D / 2):D]),axis=0).astype(np.int32))
# mask = np.zeros((nNx,nNy))
#
# mask = mask.astype(np.bool)
# distArray = ndi.distance_transform_edt(~mask)
#
# max_array = np.max(,axis=0)





