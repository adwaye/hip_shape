
import os.path

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

plane = Plane.best_fit(Points(template_points))

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111,projection='3d')

ax.set_box_aspect(aspect=(1,1,1))
ax.plot(template_points[:,0],template_points[:,1],template_points[:,2],color='blue')
lims = np.min(template_points,axis=0),np.max(template_points,axis=0)
#plane.plot_3d(ax,lims_y=(-150,50),lims_x=(-75,50),alpha=0.2)

k=0
for pt in template_points:
    pt_proj = plane.project_point(pt)
    alpha   = -pt_proj[2]/plane.normal[2]
    pt_xy   = pt_proj + alpha*plane.normal
    ax.scatter(pt_xy[0],pt_xy[1],pt_xy[2],color='red')
    if k ==0:
        pts_xy = np.expand_dims(pt_xy,axis=0)
    else:
        pt_xy_ = np.expand_dims(pt_xy,axis=0)
        pts_xy  = np.concatenate((pts_xy,pt_xy_),axis=0)
    k+=1


x = pts_xy[:,0]
y = pts_xy[:,1]

fig,ax=plt.subplots()
ax.plot(x,y,'o')
labels = ['{0}'.format(i) for i in range(len(x))]
skips = 1
for label,x_,y_ in zip(labels[::skips],x[::skips],y[::skips]):
    ax.annotate(label,xy=(x_,y_),xytext=(-10,10),textcoords='offset points',ha='right',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',fc='yellow',alpha=0.1),
                arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0'))

fourier_fit = FourierFitter(x,y)
#fixme: this is not working with fourier as shape is not closed. instead use a curve fitter as it is one-one
t = np.arange(0,10,1)/10
t = t.reshape(t.shape[0],1,1)

sampled_pts = fourier_fit.sample_pts(t)

ax.plot(sampled_pts[:,0,0],sampled_pts[:,0,1])







