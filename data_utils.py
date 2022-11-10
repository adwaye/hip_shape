
import os.path

import plotly.graph_objects
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from read_landmarks import *
from datetime import datetime
import pickle

import numpy as np
from stl import mesh  # pip install numpy-stl
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
import threading as t


source_loc = './data/Segmentation_and_landmarks_raw/'
target_loc = './data/Segmentation_and_landmarks_processed/'
APP_aligned_loc = './data/Segmentation_and_landmarks_APP_aligned/'
RSocket_aligned_trans = './data/Segmentation_and_landmarks_Rsocket_aligned/'
LSocket_aligned_trans = './data/Segmentation_and_landmarks_Lsocket_aligned/'
dtype = jnp.float32




class HipData(object):
    def __init__(self,pickle_path):
        self.pickle_path = pickle_path
        with open(pickle_path,'rb') as fp:
            data = pickle.load(fp)
        self.data   = data
        # try:
        #     self.rot_mat    = self.data['rotmat']
        # except KeyError:
        #     self.rot_mat = None
        #
        # try:
        #     self.trans_vect = self.data['trans_vect']
        # except KeyError:
        #     self.trans_vect = None


    def rotate(self,points):
        """

        :param points: points to be rotated
        :type points: array shape (3,N)
        :return: rotated points
        :rtype: array shape (3,N)
        """
        if points.shape[0]!=3:
            points = np.transpose(points)
        if self.rot_mat is None:
            print('no rotation matrix found in data')
        else:
            center  = np.mean(points,axis=1,keepdims=True)
            points_ = points - np.mean(points,axis=1,keepdims=True)
            points  = np.matmul(self.rot_mat,points_)+center
        return points.transpose()

    def translate(self,points):
        """

        :param points: points to be rotated
        :type points: array shape (3,N)
        :return: translated points
        :rtype: array shape (3,N)
        """

        if self.trans_vect is None:
            print('no translation vector found in data')
        else:
            points = points + self.trans_vect
        return points

    @property
    def APP_coords(self):
        """returns the app plane coodinates

        :return: array shape (4,3) containing the APP coordinates
        :rtype:np.array
        """
        points = None
        k = 0
        for key in ['RASIS','LASIS','RTUB','LTUB']:
            try:
                if k == 0:
                    points = np.expand_dims(self.data['landmarks'][key],axis=0)
                else:
                    points_ = np.expand_dims(self.data['landmarks'][key],axis=0)
                    points = np.concatenate((points,points_),axis=0)
                k+=1
            except KeyError:
                print(f'cannot find {key}')

        if points is not None:
            points = self.rotate(points)
            points = self.translate(points)

        return points


    @property
    def RPEL(self):
        """returns the right pelvis point cloud. if self.rot_mat is not None, the points are rotated.
        Similarly if self.trans_vect is not None, the points are translated

        :return: points->reshaped vertices shape [n_vertice*3,3] faces corresponding to the mesh cloud attributes
        :rtype: list
        """
        try:
            points = self.data['surface']['RPel']['points']
            faces  = self.data['surface']['RPel']['faces']
        except KeyError:
            points = None
            faces  = None
        if points is not None:
            points = self.rotate(points)
            points = self.translate(points)
        return points,faces
    @property
    def LPEL(self):
        """returns the left pelvis point cloud. if self.rot_mat is not None, the points are rotated.
        Similarly if self.trans_vect is not None, the points are translated

        :return:
        :rtype:
        """
        try:
            points = self.data['surface']['LPel']['points']
            faces = self.data['surface']['LPel']['faces']
        except KeyError:
            points = None
            faces  = None
        if points is not None:
            points = self.rotate(points)
            points = self.translate(points)
        return points,faces
    @property
    def right_socket(self):
        """returns the right socket plane crest points. if self.rot_mat is not None, the points are rotated.
        Similarly if self.trans_vect is not None, the points are translated

        :return: points correponding to the socket opening plane
        :rtype: np.array
        """

        for side in ['Right']:  #['Right','Left']:
            k = 0
            for key in ['Ant Lat','Post Lat']:
                try:
                    if k == 0:
                        points = self.data['landmarks'][side + ' ' + key]
                    else:
                        points_ = self.data['landmarks'][side + ' ' + key]

                        points = np.concatenate((points,points_),axis=0)
                    k += 1
                except KeyError:
                    print('cannot find ' + side + ' ' + key)

        points = self.rotate(points)
        points = self.translate(points)
        return points

    @property
    def left_socket(self):
        """returns the left socket plane crest points. if self.rot_mat is not None, the points are rotated.
        Similarly if self.trans_vect is not None, the points are translated

        :return: points correponding to the socket opening plane
        :rtype: np.array
        """
        for side in ['Left']:  #['Right','Left']:
            k = 0
            for key in ['Ant Lat','Post Lat']:
                try:
                    if k == 0:
                        points = self.data['landmarks'][side + ' ' + key]
                    else:
                        points_ = self.data['landmarks'][side + ' ' + key]

                        points = np.concatenate((points,points_),axis=0)
                    k += 1
                except KeyError:
                    print('cannot find ' + side + ' ' + key)
        points = self.rotate(points)
        points = self.translate(points)
        return points

    @property
    def rot_mat(self):
        try:
            rot_mat = self.data['rotmat']
        except KeyError:
            rot_mat = None
        return rot_mat


    @rot_mat.setter
    def rot_mat(self,val):
        assert val.shape==(3,3)
        self.data['rotmat'] = val
#        self.rot_mat = val

    @property
    def trans_vect(self):
        try:
            trans_vect = self.data['translation']
        except KeyError:
            trans_vect = None
        return trans_vect

    @trans_vect.setter
    def trans_vect(self,val):
        assert val.shape == (1,3)
        self.data['translation'] = val
        #self.rot_mat = val

    def save_data(self,location):
        """Saves self.data as a pickle file. The saved file name is the same as the original filename. The pickle
        file is saved in location/self.pickle_path

        :param location: path pointing to a folder where the pickle file should saved
        :type location: str
        :return:
        :rtype:
        """
        file_path = os.path.join(location,self.pickle_path.split('/')[-1])
        with open(file_path,'wb') as fp:
            pickle.dump(out_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)


    def get_plotly_graph(self,color: str='lightpink')->list[plotly.graph_objects.mesh3d]:
        """Returns the plotly 3d graph associated with each of the left and right pelvis

        :return: [left_mesh_plot,right_mesh_plot] left and right pelvis pot objects
        :rtype: tuple(plotly.graph_objects.Mesh3d,plotly.graph_objects.Mesh3d)
        """
        left_mesh_plot  = None
        right_mesh_plot = None


        r_mesh, _ = self.RPEL
        l_mesh, _ = self.LPEL
        if r_mesh is not None:
            vertices,I,J,K = points2mesh3d(r_mesh)#-np.mean(self.RPEL[0],axis=0,keepdims=True))
            x,y,z = vertices.T
            right_mesh_plot = go.Mesh3d(x=x
                                             ,y=y
                                             ,z=z
                                             ,i=I
                                             ,j=J
                                             ,k=K
                                             ,color=color,opacity=0.50)
        if l_mesh is not None:
            vertices,I,J,K = points2mesh3d(l_mesh)#-np.mean(self.RPEL[0],axis=0,keepdims=True))
            x,y,z = vertices.T
            left_mesh_plot = go.Mesh3d(x=x
                                             ,y=y
                                             ,z=z
                                             ,i=I
                                             ,j=J
                                             ,k=K
                                             ,color=color,opacity=0.50)
        return left_mesh_plot,right_mesh_plot






def _extract_data():
    """scipt that extracts the right socket from two patient files

    :return: template_points, target_points which represent the socket planes of 2 different patient
    :rtype: tuple
    """
    dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")



    paths         = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']
    location      = os.path.join(target_loc,paths[0])
    files         = [os.path.join(location,f) for f in os.listdir(location)]
    template_file = files[11]
    target_file   = files[15]

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


def _extract_data_cloud():
    """scipt that extracts the right socket from two patient files

    :return: (template_points,template_data['surface']['RPel']), (target_points,target_data['surface']['RPel'])
            template_points, target_points: poiints of the template, target socket plane,
            template_data['surface']['RPel'], target_data['surface']['RPel'] template, Target,m dictionary with keys
            'faces', 'points', 'mesh_loc'
            containing the
            Right
            pelvis surface
    :rtype: list of tuples
    """
    dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")



    paths         = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']
    location      = os.path.join(target_loc,paths[0])
    files         = [os.path.join(location,f) for f in os.listdir(location)]
    template_file = files[11]
    target_file   = files[15]

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
    return (template_points,template_data),(target_points,target_data)


def umeyama(P,Q):
    """- http://stackoverflow.com/a/32244818/263061 (solution with scale)
      - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)

    Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    with least-squares error.
    Returns (scale factor c, rotation matrix R, translation vector t) such that
      Q = P*cR + t
    if they align perfectly, or such that
      SUM over point i ( | P_i*cR + t - Q_i |^2 )
    is minimised if they don't align perfectly.

    :param P:
    :type P:
    :param Q:
    :type Q:
    :return:
    :rtype:
    """
    assert P.shape == Q.shape
    n,dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP),centeredQ) / n

    V,S,W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    R = np.dot(V,W)

    varP = np.var(P,axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

    return c,R,t





def rotation_between_vectors(normal1,normal2):
    """Returns the rotation matrix that maps vector normal2 onto vector normal1

    :param normal1: 3d vector shape (3,)
    :type normal1: jnp.array
    :param normal2: 3d vector shape (3,)
    :type normal2: jnp.array
    :return: rotation matrix mapping normal2 onto normal1 shape (3,3)
    :rtype: jnp.array

    >>> normal1 = jnp.array(np.random.randn(3))
    >>> normal1 /= jnp.sqrt(jnp.dot(normal1,normal1))
    >>> normal2 = jnp.array(np.random.randn(3))
    >>> normal2 /= jnp.sqrt(jnp.dot(normal2,normal2))
    >>> rot_mat = rotation_between_vectors(normal1,normal2)
    >>> normal2_trans = jnp.matmul(rot_mat,normal2)
    >>> print(jnp.sum(jnp.square(normal1-normal2_trans)))

    """
    normal1 /= jnp.sqrt(jnp.dot(normal1,normal1))
    normal2 /= jnp.sqrt(jnp.dot(normal2,normal2))

    cosine      = jnp.dot(normal1,normal2)/(jnp.sqrt(jnp.dot(normal1,normal1))*jnp.sqrt(jnp.dot(normal2,normal2)))
    sine        = -jnp.sqrt(1-jnp.square(cosine))
    u = np.cross(normal1,normal2)
    u = u/jnp.sqrt(jnp.dot(u,u))

    rot_mat = jnp.array([
                        [cosine+u[0]**2*(1-cosine), u[0]*u[1]*(1-cosine) - u[2]*sine, u[0]*u[2]*(1-cosine)+u[1]*sine],
                        [u[1]*u[0]*(1-cosine)+u[2]*sine, cosine + u[1]**2*(1-cosine), u[1]*u[2]*(1-cosine)-u[0]*sine],
                        [u[2]*u[0]*(1-cosine)-u[1]*sine, u[2]*u[1]*(1-cosine)+u[0]*sine,cosine + u[2]**2*(1-cosine)]
                             ],dtype=dtype)

    return rot_mat

def ralign_2_hips(hip1: HipData,hip2: HipData,by: str='RPel')->list[HipData]:
    """Performs rigid alignment of two hips.

    :param hip1:
    :type hip1: HipData
    :param hip2:
    :type hip2: HipData
    :param by: options are

    :type by: str
    :return:
    :rtype:
    """

    if by.lower()=='rpel':
        print('aligning right pelves')

        template_points = hip1.right_socket
        target_points = hip2.right_socket
        template_plane = Plane.best_fit(Points(template_points))
        target_plane = Plane.best_fit(Points(target_points))

        rot_mat = rotation_between_vectors(template_plane.normal,target_plane.normal)
        mean2 = -np.mean(hip2.RPEL[0],axis=0,keepdims=True)#translation hip1
        mean1 = -np.mean(hip1.RPEL[0],axis=0,keepdims=True)#translation hip2


    elif by.lower() == 'lpel':
        print('aligning left pelves')

        template_points = hip1.left_socket
        target_points = hip2.left_socket
        template_plane = Plane.best_fit(Points(template_points))
        target_plane = Plane.best_fit(Points(target_points))

        rot_mat = rotation_between_vectors(template_plane.normal,target_plane.normal)
        mean2 = -np.mean(hip2.LPEL[0],axis=0,keepdims=True)
        mean1 = -np.mean(hip1.LPEL[0],axis=0,keepdims=True)

    else:
        print('aligning the whole thing')

        template_points = hip1.APP_coords
        target_points = hip2.APP_coords
        # template_plane = Plane.best_fit(Points(template_points))
        # target_plane = Plane.best_fit(Points(target_points))
        c,R,t = umeyama(target_points,template_points)

#        rot_mat = rotation_between_vectors(template_plane.normal,target_plane.normal)
        rot_mat = R
        mean2 = -np.mean(np.concatenate((hip2.LPEL[0],hip2.RPEL[0]),axis=0),axis=0,keepdims=True)
        mean1 = -np.mean(np.concatenate((hip1.LPEL[0],hip1.RPEL[0]),axis=0),axis=0,keepdims=True)

    hip1.trans_vect = mean1
    hip2.rot_mat    = rot_mat
    hip2.trans_vect = mean2

    return hip1,hip2


def loadSTL(filename):
    m = mesh.Mesh.from_file(filename)
    shape = m.points.shape
    points = m.points.reshape(-1,3)
    print(points.shape)
    faces = np.arange(points.shape[0]).reshape(-1,3)
    return points,faces


def stl2mesh3d(stl_mesh):
    """

    :param stl_mesh:
    :type stl_mesh:
    :return:
    :rtype:
    """
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K

def points2mesh3d(points):
    """Turns a point cloud surface into a format that is plottable by plotly, the input should be the points
    attribute to a numpy.stl loaded mesh file. Alternatively, points need to be loaded from the pickle output of
    DataCleaning._to_pickle

    :param points: array shape [npoints*3,3]. points attribute of a mesh object or
    :type points: array
    :return: vertices,I,J,K
    :rtype: list
    """
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    points = points.reshape(points.shape[0]//3,3,3)
    p, q, r = points.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(points.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K



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


    points = jnp.array([1,1,1],dtype=dtype)
    angle  = jnp.pi/2
    Rx = jax_roll_matrix(angle)
    Ry = jax_pitch_matrix(angle)
    Rz = jax_yaw_matrix(angle)
    rot_matrix = jax_rotation_matrix3d(angle,angle,angle)
    assert jnp.linalg.norm(jnp.matmul(Rx,points) - jnp.array([1,-1,1],dtype=dtype))<10e-5
    assert jnp.linalg.norm(jnp.matmul(Ry,points) - jnp.array([1,1,-1],dtype=dtype))<10e-5
    assert jnp.linalg.norm(jnp.matmul(Rz,points) - jnp.array([-1,1,1],dtype=dtype))<10e-5
    assert jnp.linalg.norm(rot_matrix-jnp.matmul(Rz,jnp.matmul(Ry,Rx)))<10e-5
    rot_matrix = jax_rotation_matrix3d(0.0,0.0,0.0)

    print(f'Rotation matrix size {rot_matrix.shape}')
    print(f'points  size {template_points_proc.shape}')
    print(f'points transpose size {template_points_proc.transpose().shape}')

    assert jnp.linalg.norm(jnp.matmul(rot_matrix,template_points_proc.transpose())-template_points_proc.transpose())<10e-5

    print(jnp.matmul(rot_matrix,points))

    params = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0],dtype=dtype)
    print('evaluating distance')
    loss = affine_cost(params,template_points_proc,dt)
    assert loss<10-3





def _test_vector_rotation():
    normal1 = jnp.array(np.random.randn(3))
    normal1 /= jnp.sqrt(jnp.dot(normal1,normal1))
    normal2 = jnp.array(np.random.randn(3))
    normal2 /= jnp.sqrt(jnp.dot(normal2,normal2))
    rot_mat = rotation_between_vectors(normal1,normal2)
    normal2_mapped = jnp.matmul(rot_mat,normal2)
    print(f'========Lin-alg=================')
    print(f'mapped normal {normal2_mapped}')
    print(f'original normal {normal1}')
    assert jnp.linalg.norm(normal2_mapped-normal1)<10-4












#need to embed the curves into an array:
#calculating size of the array to allocate for the distance transform
if __name__=='__main__':


    template_shape = HipData('/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - '
                        'Controls/C4.p')
    mean_pos=np.mean(template_shape.RPEL[0],axis=0,keepdims=True)
    print(template_shape.APP_coords.shape)
    #template_shape.trans_vect = np.mean(template_shape.RPEL[0],axis=0,keepdims=True)
    target_shape = HipData('/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - '
                        'Controls/C8.p')
    #target_shape.trans_vect = np.mean(target_shape.RPEL[0],axis=0,keepdims=True)

    template_shape,target_shape = ralign_2_hips(template_shape,target_shape,by='all')

    # template_points = template_shape.right_socket
    # target_points = target_shape.right_socket
    #
    # template_plane = Plane.best_fit(Points(template_points))
    # target_plane = Plane.best_fit(Points(target_points))
    #
    #
    # rot_mat = rotation_between_vectors(template_plane.normal,target_plane.normal)
    #
    #
    # target_shape.rot_mat = rot_mat
    # target_shape.trans_vect = -np.mean(target_shape.RPEL[0],axis=0,keepdims=True)
    # template_shape.trans_vect = -np.mean(template_shape.RPEL[0],axis=0,keepdims=True)

    left_plot2,right_plot2   = target_shape.get_plotly_graph()
    left_plot1,right_plot1 = template_shape.get_plotly_graph(color='blue')
    fig =  go.Figure(data=[left_plot1
                          ,left_plot2
                          ,right_plot1
                          ,right_plot2
                          ])

    fig.show()





