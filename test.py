"""Test module to test jax optimisation loop with a mayavi visualiser for the hips. IN this script, the jax loop
is a simple sequential rotation. Mayavi visualiser can be updated as long as it is fed a generator. See
https://stackoverflow.com/questions/39840638/update-mayavi-plot-in-loop and https://docs.enthought.com/mayavi/mayavi/mlab_animating.html for more info.

"""

from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math
from visualiser import HipDataSource, MayaviVisualiser
import os
from tvtk.api import tvtk
import jax
import jax.numpy as jnp
from utils import jax_yaw_matrix
import sys
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
plt.ion()
# ax = plt.axes(projection='3d',label='i')
# ax.scatter(YY_[0,:,0],YY_[0,:,1],YY_[0,:,2])
#
# mesh = ConvexHull(YY_)

if __name__=='__main__':
    alpha = np.linspace(0, 2*math.pi, 100)
    key = jax.random.PRNGKey(1234)

    # xs = np.cos(alpha)
    # ys = np.sin(alpha)
    # zs = np.zeros_like(xs)
    #
    # mlab.points3d(0,0,0)
    # plt = mlab.points3d(xs[:1], ys[:1], zs[:1])








    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    # points,faces = hip_data_source.get_data()
    # polydata = tvtk.PolyData(points=points,polys=faces)
    #
    # mesh = mlab.pipeline.surface(polydata)


    def jax_func(data):
        noise = jax.random.normal(key=key,shape=data.shape)
        noise = jnp.multiply(100,noise)

        return jnp.add(noise,data)

    @jax.jit
    def rotate(points,rot_matrix):
        return jnp.matmul(rot_matrix,points)



    def generator(hip_data_source):
        rot_matrix = jax_yaw_matrix(angle=np.pi/30)
        points , _ = hip_data_source.get_data()
        points = jnp.transpose(points)
        for k in range(100):


            # points = np.array(jnp.matmul(rot_matrix,points))
            points = rotate(points,rot_matrix)
            print('rotating')


            yield np.array(jnp.transpose(points)),_
        #return  np.array(jnp.transpose(points))
    #fixme: you cannot jit a generator




    file_gen = generator(hip_data_source=hip_data_source)
    mayavi_visualiser =MayaviVisualiser(generator=file_gen)
    points,faces = hip_data_source.get_data()
    # mayavi_visualiser.polydata = tvtk.PolyData(points=points,polys=faces)
    # mayavi_visualiser.mesh = mlab.pipeline.surface(mayavi_visualiser.polydata)

    mayavi_visualiser.set_data(hip_data_source.get_data())
    mayavi_visualiser.display_data()
    mayavi_visualiser.update_visualisation()


    # @mlab.animate(delay=500)
    # def anim():
    #     while True:
    #         try:
    #             points,faces = next(file_gen)
    #         except StopIteration:
    #              break
    #         mayavi_visualiser.polydata.points = points
    #         mayavi_visualiser.polydata.polys  = faces
    #         mayavi_visualiser.mesh.parent.parent.update()
    #         yield
    # #
    # #
    # #
    # anim()
    mlab.show()
    # for k in range(10):
    #     points,_=next(file_gen)
    #     print(points)
    #mlab.show()