
import os.path



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

source_loc = './data/Segmentation_and_landmarks_raw/'
target_loc = './data/Segmentation_and_landmarks_processed/'
APP_aligned_loc = './data/Segmentation_and_landmarks_APP_aligned/'
Socket_aligned_trans = './data/Segmentation_and_landmarks_socket_aligned/'
dtype = jnp.float64




class HipData(object):
    def __init__(self,pickle_path):
        self.pickle_path = pickle_path
        with open(pickle_path,'rb') as fp:
            data = pickle.load(fp)
        self.data   = data
        try:
            self.rot_mat    = self.data['rotmat']
            self.trans_vect = self.data['translation']
        except KeyError:
            self.rot_mat = None
            self.trans_vect = None


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
                    points = self.data['landmarks'][key]
                else:
                    points_ = self.data['landmarks'][key]
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
    def rot_mat(self):
        return self.rot_mat


    @rot_mat.setter
    def rot_mat(self,val):
        self.data['rotmat'] = val
        self.rot_mat = val

    @property
    def trans_vect(self):
        return self.trans_vect

    @trans_vect.setter
    def trans_vect(self,val):
        self.data['translation'] = val
        self.rot_mat = val

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



def align_2_sockets(socket1,socket2):
    pass


#need to embed the curves into an array:
#calculating size of the array to allocate for the distance transform
if __name__=='__main__':
    # _test_modules()
    #_test_vector_rotation()

    template_tup, target_tup = _extract_data_cloud()
    template_points = template_tup[0]
    target_points   = target_tup[0]
    template_data   = template_tup[1]
    target_data     = target_tup[1]


    template_plane = Plane.best_fit(Points(template_points))
    target_plane = Plane.best_fit(Points(target_points))


    rot_mat = rotation_between_vectors(template_plane.normal,target_plane.normal)

    template_surface = jnp.array(template_data['surface']['RPel']['points'])
    template_surface = template_surface-jnp.mean(template_surface,axis=0,keepdims=True)
    target_surface   = jnp.array(target_data['surface']['RPel']['points']).transpose()

    target_surface_trans = jnp.matmul(rot_mat,target_surface-jnp.mean(target_surface,axis=1,keepdims=True))
    #target_surface_trans = jnp.subtract(target_surface_trans,jnp.mean(template_surface,axis=0,keepdims=True))

    import plotly.graph_objects as go

    import plotly.io as pio

    pio.renderers.default = "browser"
    pio.renderers



    vertices,I,J,K = points2mesh3d(template_surface)#todo: extract this data for the target mesh as well: the rotmat
    _vertices,_I,_J,_K = points2mesh3d(target_surface_trans.transpose())
    # needs
    # to act on the vertices
    x,y,z = vertices.T
    mesh_plot = go.Mesh3d(x=x
                                    ,y=y
                                    ,z=z
                                    ,i=I
                                    ,j=J
                                    ,k=K
                                    ,color='lightpink',opacity=0.50)
    x,y,z = _vertices.T
    mesh_plot_ = go.Mesh3d(x=x
                                    ,y=y
                                    ,z=z
                                    ,i=_I
                                    ,j=_J
                                    ,k=_K
                                    ,color='blue',opacity=0.50)
    fig = go.Figure(data=[mesh_plot
                         ,mesh_plot_
                          ])
    fig.show()



    #
    # skip = 20
    # fig = plt.figure(figsize=(4,4))
    #
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot_surface(template_surface[:,0],template_surface[:,1],template_surface[:,2])
    # plt.show()
    #
    # ax.set_box_aspect(aspect=(1,1,1))
    # # ax.plot(template_points_proc[:,0],template_points_proc[:,1],template_points_proc[:,2],color='blue',label='Socket plane points')
    #
    # ax.scatter(template_surface[::skip,0],template_surface[::skip,1],template_surface[::skip,2],color='blue',alpha=0.5,
    #         label='Socket plane points')
    # ax.scatter(target_surface_trans[0,::skip],target_surface_trans[1,::skip],target_surface_trans[2,::skip],
    #            color='red',
    #            alpha=0.5,
    #            label='Socket plane points')
    #
    # plt.show()






    # from jax_transformations3d import jax_transformations3d as jts
    #
    #
    # template_points,target_points = _extract_data()
    # template_points_mean  = np.mean(template_points,axis=0,keepdims=True)
    # target_points_mean = np.mean(template_points,axis=0,keepdims=True)
    # template_points -= template_points_mean
    # target_points -= target_points_mean
    #
    # template_plane = Plane.best_fit(Points(template_points))
    # target_plane = Plane.best_fit(Points(target_points))
    #
    # normal1 = jnp.array(np.random.randn(3))
    # normal1 /= jnp.sqrt(jnp.dot(normal1,normal1))
    # normal2 = jnp.array(np.random.randn(3))
    # normal2 /= jnp.sqrt(jnp.dot(normal2,normal2))
    #
    # rot_mat = rotation_between_vectors(normal1,normal2)
    #
    # # params = np.array([0.0,0.0,0.0])
    # #
    # # def cost_function(params,normal1,normal2):
    # #     rot_matrix = jax_rotation_matrix3d(params[0],params[1],params[2])
    # #     cost = jnp.sum(jnp.square(normal1-jnp.matmul(rot_matrix,normal2)))
    # #     return cost
    # #
    # #
    # # from functools import partial
    # # # loss_fn = partial(cost_function,normal1=normal1,normal2=normal2)
    # # loss_fn = lambda z: cost_function(z, normal1, normal2)
    # #
    # #
    # # res = optimize.minimize(loss_fn,params,method='BFGS',options={'maxiter':100})
    # #
    # # rot_matrix = jax_rotation_matrix3d(*res.x)
    # rot_matrix = align_Planes_bfgs(template_plane,target_plane)
    # normal2_mapped = jnp.matmul(rot_matrix,normal2)
    # print(f'========BFGS=================')
    # print(f'maped nnormal {normal2_mapped}')
    # print(f'original nnormal {normal1}')
    #
    # normal2_mapped = jnp.matmul(rot_mat,normal2)
    # print(f'========Lin-alg=================')
    # print(f'maped nnormal {normal2_mapped}')
    # print(f'original nnormal {normal1}')





























    # template_points_proc,target_points_proc,size = _embed_points_in_array(template_points,target_points,padding=(300,
    #                                                                                                              300))
    # dt = _make_distance_transform(points=template_points_proc,size=size,ncpus=2)
    #
    #
    # from functools import partial
    # method = 'BFGS'
    # nits =100
    # loss_fun = partial(affine_cost,points=target_points_proc,dist=dt)

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
    # params = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0],dtype=dtype)
    # #
    # res = optimize.minimize(loss_fun,params,method=method,options=options)#,callback=callback_function)
    # new_params = res.x
    # rot_matrix = jax_rotation_matrix3d(new_params[0],new_params[1],new_params[2])
    # trans_points = jnp.add(jnp.matmul(rot_matrix,target_points_proc.transpose()),jnp.expand_dims(new_params[3:],axis=1))
    # rot_points   = jnp.matmul(rot_matrix,target_points_proc.transpose())
    # point_plot = ax.scatter(trans_points[0,:],trans_points[1,:],trans_points[2,:],color='black',
    #                      label='Transformed points')




    #test_modules()









