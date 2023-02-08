
from pyqtgraph.Qt import QtCore,QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
from stl import mesh

from pathlib import Path


class MyWindow(QMainWindow):
    """QMainWindow class that allows one to load and view an stl file. The displayed stl is centered before showing.

    """
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setGeometry(0,0,700,900)
        self.setAcceptDrops(True)

        self.initUI()

        self.currentSTL = None
        self.lastDir = None

        self.droppedFilename = None

    def initUI(self):
        """initialises the widgets in the UI

        :return:
        :rtype:
        """
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)

        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer,1)

        self.viewer.setWindowTitle('STL Viewer')
        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(200,200)
        g.setSpacing(5,5)
        #self.viewer.addItem(g)

        btn = QPushButton(text="Load STL")
        btn.clicked.connect(self.showDialog)
        btn.setFont(QFont("Ricty Diminished",14))
        layout.addWidget(btn)

    def showDialog(self):
        """Shows the file selection dialog to choose the stl file from.

        :return:
        :rtype:
        """
        directory = Path("./data")
        if self.lastDir:
            directory = self.lastDir
        fname = QFileDialog.getOpenFileName(self,"Open file",'./data',"STL (*.stl)")
        if fname[0]:
            self.showSTL(fname[0])
            self.lastDir = Path(fname[0]).parent

    def showSTL(self,filename):
        """Loads the stl file from the filename. Gets called by clicking ok on the file dialog

        :param filename: path to stl file
        :type filename:  str
        :return:
        :rtype:
        """
        if self.currentSTL:
            self.viewer.removeItem(self.currentSTL)

        points,faces = self.loadSTL(filename)
        meshdata = gl.MeshData(vertexes=points,faces=faces)
        mesh = gl.GLMeshItem(meshdata=meshdata,smooth=True,drawFaces=True,drawEdges=False,edgeColor=(0,1,0,1),
                             shader='shaded')
        self.viewer.addItem(mesh)
        mean_pos = np.mean(points,axis=0)
        self.viewer.pan(mean_pos[0],mean_pos[1],mean_pos[2],relative='global')

        self.currentSTL = mesh

    def loadSTL(self,filename):
        m = mesh.Mesh.from_file(filename)
        shape = m.points.shape
        points = m.points.reshape(-1,3)
        print(points.shape)
        faces = np.arange(points.shape[0]).reshape(-1,3)
        return points,faces

    def dragEnterEvent(self,e):
        print("enter")
        mimeData = e.mimeData()
        mimeList = mimeData.formats()
        filename = None

        if "text/uri-list" in mimeList:
            filename = mimeData.data("text/uri-list")
            filename = str(filename,encoding="utf-8")
            filename = filename.replace("file:///","").replace("\r\n","").replace("%20"," ")
            filename = Path(filename)

        if filename.exists() and filename.suffix == ".stl":
            e.accept()
            self.droppedFilename = filename
        else:
            e.ignore()
            self.droppedFilename = None

    def dropEvent(self,e):
        if self.droppedFilename:
            self.showSTL(self.droppedFilename)


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()