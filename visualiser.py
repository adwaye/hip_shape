import jax.numpy as jnp
import jax
from tvtk.api import tvtk
import os
from traits.api import HasTraits,Instance,on_trait_change
from traitsui.api import View,Item
from mayavi.core.ui.api import MayaviScene,MlabSceneModel,SceneEditor
import pyqtgraph

from data_utils import HipData
import numpy as np
from pyqtgraph.Qt import QtCore,QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

import threading

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()


from mayavi import mlab
from mayavi.api import Engine
import time
class Observer(object):
    def __init__(self):
        pass

    def __eq__(self, other):
        """needs to be overridden so that the subject (DataSource) can remove the observer

        :param other: variable to be compared
        :type other: object
        :return: True if equal and False if not
        :rtype: bool
        """
        if isinstance(other,Observer):
            return True
        else:
            return False

    def set_data(self,data:list):
        """method to read in the data

        :param data:data[0] should be vertices, and data[1] should be connectivity matrix of vertices,
        :type data:
        :return:
        :rtype:
        """
        self.points = data[0]
        self.faces  = data[1]




    def display_data(self):
        pass



class GlObserver(Observer,gl.GLViewWidget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        gl.GLViewWidget.__init__(self,**kwargs)
        self.mesh_data = None

    def display_data(self):
        #self.clear()
        points = self.points - np.mean(self.points,axis=0)
        faces = self.faces
        mean_pos = np.mean(points,axis=0)
        points = points - mean_pos

        #        self.viewer1.updateGL()
        print('mean osition of object is')
        print(mean_pos)
        print('camera position is')
        print(self.cameraPosition())
        if self.mesh_data is None:


            meshdata = gl.MeshData(vertexes=points,faces=faces)
            self.mesh_item = gl.GLMeshItem(meshdata=meshdata,smooth=True,drawFaces=True,drawEdges=False,edgeColor=(0,1,
                                                                                                                0,1),
                                 shader='shaded')
            self.mesh_data = meshdata

            self.addItem(self.mesh_item)
            self.pan(mean_pos[0],mean_pos[1],mean_pos[2],relative='global')
        else:
            self.mesh_data.setFaces(faces)
            self.mesh_data.setVertexes(points)
            self.mesh_item.setMeshData(meshdata=self.mesh_data)
            self.mesh_item.meshDataChanged()
            self.mesh_item.update()

        self.update()


class MatplotlibObserver(Observer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        #HasTraits.__init__(self,**kwargs)
        self.tri_plot= None
    def set_data(self,data:list):

        super().set_data(data)

    def display_data(self):

        ## PLot to Show
        if self.tri_plot is None:
            self.init_mesh()
            plt.show()
        else:
            self.update_visualisation()

    def init_mesh(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        vertices = self.points
        faces = self.faces
        self.tri_plot = self.ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces,edgecolor=[[0,0,
                                                                                                                 0]],
                                   linewidth=1.0,
                                   alpha=0.0,
                                   shade=False)
        plt.pause(0.01)
        plt.draw()

    def update_visualisation(self):
#        pass

        self.tri_plot.set_verts_and_codes(verts=self.points,codes=self.faces)
        plt.pause(0.01)
        # self.fig.canvas.draw()
        # self.tri_plot.update()

class MayaviObserver(Observer,HasTraits):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        HasTraits.__init__(self,**kwargs)
        self.view = View(Item('scene',editor=SceneEditor(scene_class=MayaviScene),
                     height=250,width=300,show_label=False),
                resizable=True)
        self.engine = Engine()
        self.engine.start()
        self.engine.new_scene()
        self.mesh = None
        self.polydata = None
    def set_data(self,data:list):

        super().set_data(data)

        #self.mesh.mlab_source.set()

    def display_data(self):

        ## PLot to Show
        if self.mesh is None:
            self.init_mesh()
            mlab.show()
        else:
            self.update_visualisation()



        # x = my_mesh.x
        # y = my_mesh.y
        # z = my_mesh.z
        # mlab.points3d(x, y, z)
        # mlab.show()
        # color = [(139 / 255,233 / 255,253 / 255),(80 / 255,250 / 255,123 / 255),(139 / 255,233 / 255,253 / 255),
        #          (80 / 255,250 / 255,123 / 255)]
        #
        # if self.plot is None:
        #     self.mesh_pipeline = mlab.pipeline.triangular_mesh_source(self.points[:,0],self.points[:,1],self.points[:,2],self.faces)
        #     self.plot = mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2],self.faces)
        # else:
        #     self.plot.mlab
        #
        #     mlab.show()
        # self.mesh = mlab.triangular_mesh()


    def init_mesh(self):
        self.polydata = tvtk.PolyData(points=self.points,polys=self.faces)


        self.mesh = mlab.pipeline.surface(self.polydata)


    @mlab.animate(delay=500,ui=False)
    def update_visualisation(self):
        pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - ' \
                     'Controls/'
        files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]

        # hip_data_source.set_data(pickle_path=pickle_path)
        # points,faces = hip_data_source.get_data()
        # points = points - np.mean(points,axis=0,keepdims=True)
        self.polydata.points = self.points
        self.polydata.polys = self.faces


        # drawing.mlab_source.x = x
        # drawing.mlab_source.y = y
        # drawing.mlab_source.z = z
        # drawing.mlab_source.triangles = faces
        # drawing.update()
        self.mesh.parent.parent.update()
        #drawing.mlab_source.trait_set(x=x,y=y,z=z,triangles=faces)
        yield






class DataSource(object):
    def __init__(self,**kwargs):
        self.observers = []



    def registerObserver(self,observer:Observer):
        """Registers an observer who is then updated each time the data is changed in DataSource

        :param observer:
        :type observer:
        :return:
        :rtype:
        """
        self.observers+=[observer]

    def removeObserver(self,observer):
        """Cycles the list of observers registered so far, checks which element is equal to observer and removes
        the observer

        :return:
        :rtype:
        """
        self.observers.remove(observer)

    def notifyObservers(self):
        """
        Notufy the observers that the data has changed
        """
        data = self.get_data()
        for i,obs in enumerate(self.observers):
            obs.set_data(data)
            obs.display_data()

    def get_data(self)->list[np.array]:
        """This needs to be ovewritten for each new class

        :return:
        :rtype:
        """
        pass




class HipDataSource(HipData,DataSource):
    def __init__(self,pickle_path:str,decimator=None,**kwargs):

        super().__init__(pickle_path=pickle_path,decimator=decimator)
        DataSource.__init__(self,**kwargs)

    def set_data(self,pickle_path:str):
        """Method to change the data in the

        :param data:
        :type data:
        :return:
        :rtype:
        """
        super().__init__(pickle_path=pickle_path,decimator=self.decimator)
        self.notifyObservers()

    def get_data(self) ->list[np.array]:
        """returns the surface of the Left and right pelvis as list [vertice,connectivity_matrix]

        :return:
        :rtype:
        """
        points1,faces1 = self.RPEL
        points2,faces2 = self.LPEL
        if points1 is not None:
            points = points1
            faces  = faces2
            if points2 is not None:
                points = np.concatenate([points,points2])
                faces  = np.concatenate([faces,faces2+points1.shape[0]])

        else:
            points = points2
            faces  = faces2


        return points,faces




class MyWindow(QMainWindow):
    def __init__(self,observer:Observer):
        super().__init__()
        self.observer = observer


def _test_inheritance():
    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    assert hip_data_source.observers==[]
    assert type(hip_data_source.data) is dict


def _test_data_getter():
    import time

    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    print(points,faces)



    gl_observer = MatplotlibObserver()

    #my_window = MyWindow(gl_observer)
    # gl_observer.setMinimumWidth(800)
    # gl_observer.setMinimumHeight(800)
    hip_data_source.registerObserver(observer=gl_observer)
    hip_data_source.notifyObservers()





    pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - Controls/'
    files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]
    for k,pickle_path in enumerate(files):
        time.sleep(5)
        if k<10:
            print(pickle_path)
            hip_data_source.set_data(pickle_path=pickle_path)
            hip_data_source.notifyObservers()

        else:
            break
    #sys.exit(app.exec_())


def _test_mayavi_observer():
    import time
    print(pyqtgraph.getConfigOption('useOpenGL'))
    pyqtgraph.setConfigOptions(useOpenGL= True)
    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)#modelOPtDataSource
    points,faces = hip_data_source.get_data()
    print(points,faces)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')


    m_observer = MayaviObserver()

    #my_window = MyWindow(m_observer)
    # m_observer.setMinimumWidth(800)
    # m_observer.setMinimumHeight(800)
    hip_data_source.registerObserver(observer=m_observer)
    hip_data_source.notifyObservers()




    #my_window.show()
    pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - Controls/'
    files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]
    for k,pickle_path in enumerate(files):
        time.sleep(5)
        if k<10:
            hip_data_source.set_data(pickle_path=pickle_path)
            hip_data_source.notifyObservers()

        else:
            break
    #sys.exit(app.exec_())



def jax_func(data):
    noise = jnp.array(np.random.randn(*data.shape)*10)
    data = jnp.add(data,noise)
    return data




class SafeTimedThread(threading.Thread):
    def __init__(self, thread_condition, scan_time, funct, *funct_args):
        threading.Thread.__init__(self)

        # Thread condition for the function to operate with
        self.tc = thread_condition

        # Defines the scan time the function is to be run at
        self.scan_time = scan_time

        # Function to be run
        self.run_function = funct

        # Function arguments
        self.funct_args = funct_args

    def run(self):
        for k in range(10):
            # Locks the relevant thread
            self.tc.acquire()

            # Begins timer for elapsed time calculation
            start_time = time.time()

            # Runs the function that was passed to the thread
            self.run_function(*self.funct_args)

            # Wakes up relevant threads to listen for the thread release
            self.tc.notify_all()

            # Releases thread
            self.tc.release()

            # Calculates the elapsed process time & sleep for the remainder of the scan time
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = self.scan_time - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print('Process time exceeds scan time')


def _animate_mayavi():


    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    points = points - np.mean(points,axis=0,keepdims=True)
    _lock = threading.Lock()


    # def run_main():
    #     print('Running Main Controller')
    @jax.jit
    def run_main():
        jax_func(data=points)

    def init_vis():
        # Creates a new Engine, starts it and creates a new scene
        engine = Engine()
        engine.start()
        engine.new_scene()

        # Initialise Plot
        polydata = tvtk.PolyData(points=points,polys=faces)


        mesh = mlab.pipeline.surface(polydata)

        return polydata,mesh

    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C8.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    points = points -np.mean(points,axis=0,keepdims=True)
    @mlab.animate(delay=10,ui=False)
    def update_visualisation(polydata,mesh):
        pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - ' \
                     'Controls/'
        files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]
        for k,pickle_path in enumerate(files):
            time.sleep(5)
            if k < 10:
                hip_data_source.set_data(pickle_path=pickle_path)
                points,faces = hip_data_source.get_data()
                points = points - np.mean(points,axis=0,keepdims=True)
                polydata.points = points
                polydata.polys = faces

                x = points[:,0]
                y = points[:,1]
                z = points[:,2]
                # drawing.mlab_source.x = x
                # drawing.mlab_source.y = y
                # drawing.mlab_source.z = z
                # drawing.mlab_source.triangles = faces
                # drawing.update()
                mesh.parent.parent.update()
                #drawing.mlab_source.trait_set(x=x,y=y,z=z,triangles=faces)
                yield

            else:
                break
        print('Updating Visualisation')
        # Pretend to receive data from external source


    c = threading.Condition()

    # Create display window
    dwg = init_vis()

    # Create safe timed thread for main thread and start
    main_thread = SafeTimedThread(c, 1.0, run_main).start()

    # Update using mlab animator
    vis_thread = update_visualisation(*dwg)

    mlab.show()


    #mlab.show()



if __name__=='__main__':
    _test_data_getter()
    #_test_mayavi_observer()
    #_animate_mayavi()
