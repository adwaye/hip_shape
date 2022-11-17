import os

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
        pass




    def display_data(self):
        pass



class GlObserver(Observer,gl.GLViewWidget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        gl.GLViewWidget.__init__(self,**kwargs)
        self.mesh_data = None



    def set_data(self,data:list):
        """stores the data as a list of points and faces, able to render multiple meshes due to the list
        structure of the data

        :param data:
        :type data:
        :return:
        :rtype:
        """

        self.points = data[0]
        self.faces  = data[1]

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



class GlObserver(Observer,gl.GLViewWidget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        gl.GLViewWidget.__init__(self,**kwargs)
        self.mesh_data = None

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
    print(pyqtgraph.getConfigOption('useOpenGL'))
    pyqtgraph.setConfigOptions(useOpenGL= True)
    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    print(points,faces)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')


    gl_observer = GlObserver()
    gl_observer.setMinimumWidth(800)
    gl_observer.setMinimumHeight(800)
    my_window = MyWindow(gl_observer)
    hip_data_source.registerObserver(observer=my_window.observer)
    hip_data_source.notifyObservers()




    my_window.show()
    pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - Controls/'
    files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]
    for k,pickle_path in enumerate(files):
        time.sleep(5)
        if k<10:
            hip_data_source.set_data(pickle_path=pickle_path)
            hip_data_source.notifyObservers()

        else:
            break
    sys.exit(app.exec_())








if __name__=='__main__':
    _test_data_getter()