#from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
#from PyQt5.QtWidgets import *

from pyface.qt import QtGui,QtCore

import os
import numpy as np
# from numpy import cos

from visualiser_utils import xray_selection_menu, stl2mesh3d


os.environ['ETS_TOOLKIT'] = 'qt4'
import pickle

## create Mayavi Widget and show
from pyqtgraph.Qt import QtCore,QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from stl import mesh





#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def __init__(self):
        self.output_loc = './'
        self.currentSTL = None
        self.sizeObject = QDesktopWidget().screenGeometry(-1)
        print(" Screen size : " + str(self.sizeObject.height()) + "x" + str(self.sizeObject.width()))


    def setupUi(self,MainWindow):
        ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200,200,1100,700)

        ## CENTRAL WIDGET
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        ##top-bottom layout:
        main_layout  = QHBoxLayout()

        ## left side
        self.menu1 = xray_selection_menu()
        scrollbar1 = QScrollArea(widgetResizable=True)
        scrollbar1.setMinimumHeight(200)
        scrollbar1.setMaximumHeight(200)
        scrollbar1.setWidget(self.menu1)
        splitter1 = QSplitter(orientation=Qt.Vertical)
        splitter1.addWidget(scrollbar1)

        self.viewer1 = gl.GLViewWidget()
        self.viewer1.setMinimumWidth(800)
        self.viewer1.setMinimumHeight(800)
        splitter1.addWidget(self.viewer1)

        #right side
        self.menu2 = xray_selection_menu()
        scrollbar2 = QScrollArea(widgetResizable=True)
        scrollbar2.setMinimumHeight(200)
        scrollbar2.setMaximumHeight(200)
        scrollbar2.setWidget(self.menu2)
        splitter2 = QSplitter(orientation=Qt.Vertical)
        splitter2.addWidget(scrollbar2)

        self.viewer2 = gl.GLViewWidget()
        self.viewer2.setMinimumWidth(800)
        self.viewer2.setMinimumHeight(800)
        splitter2.addWidget(self.viewer2)

        main_splitter = QSplitter(orientation=Qt.Horizontal)
        main_splitter.addWidget(splitter1)
        main_splitter.addWidget(splitter2)
        main_layout.addWidget(main_splitter)
        self.centralwidget.setLayout(main_layout)



        ## SET TEXT
#        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.connect_sub_buttons()

#     def retranslateUi(self,MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow","Simulator"))
# #        self.button_default.setText(_translate("MainWindow","Default Values"))
#         self.button_previous_data.setText(_translate("MainWindow","Previous Values"))

    def change_wd1(self):
        self.output_loc = self.menu1.wd_info.text()
        #self.image_widget.output_loc = self.output_loc
        if not os.path.isdir(self.output_loc):
            os.makedirs(self.output_loc)
        # self.xray_selection_menu.wd_info.setText(self.output_loc)
        # self.menu1.combobox_xrayid.clear()
        self.menu1.combobox_studyid.clear()
        self.display_studies1()

    def change_wd2(self):
        self.output_loc = self.menu2.wd_info.text()
        #self.image_widget.output_loc = self.output_loc
        if not os.path.isdir(self.output_loc):
            os.makedirs(self.output_loc)
        # self.xray_selection_menu.wd_info.setText(self.output_loc)
        # self.menu2.combobox_xrayid.clear()
        self.menu2.combobox_studyid.clear()
        self.display_studies2()

    def display_studies1(self):
        if not os.path.isdir(self.output_loc):
            return -1
        studies = [f for f in os.listdir(self.output_loc) if f.split('.')[-1]=='p']
        for it in studies:
            self.menu1.combobox_studyid.addItem(it)

    def display_studies2(self):
        if not os.path.isdir(self.output_loc):
            return -1
        studies = [f for f in os.listdir(self.output_loc) if f.split('.')[-1]=='p']
        for it in studies:
            self.menu2.combobox_studyid.addItem(it)

    def connect_sub_buttons(self):
        self.display_studies1()
        self.display_studies2()
        self.menu1.wd_info.textChanged.connect(self.change_wd1)
        self.menu2.wd_info.textChanged.connect(self.change_wd2)
        self.menu1.combobox_studyid.currentIndexChanged.connect(self.showSTL1)
        self.menu2.combobox_studyid.currentIndexChanged.connect(self.showSTL2)
        # self.menu1.current_study_info.textChanged.connect(self.open_study_creator)
        # self.menu1.current_file_info.textChanged.connect(self.open_xray_adder)

    def display_CT1(self):
        if self.menu1.combobox_studyid.count()>0:
            study_name = self.menu1.combobox_studyid.currentText()
            meta_loc   = os.path.join(self.output_loc,study_name)
            print(meta_loc)
            self.menu1.combobox_xrayid.clear()
            self.load_selected_xrays1()

    def display_CT2(self):
        if self.menu2.combobox_studyid.count()>0:
            study_name = self.menu2.combobox_studyid.currentText()
            meta_loc   = os.path.join(self.output_loc,study_name)
            print(meta_loc)
            self.menu2.combobox_xrayid.clear()
            self.load_selected_xrays2()


    def load_selected_xrays1(self):
        filename = os.path.join(self.menu1.wd_info.text(),self.menu1.combobox_studyid.currentText())
        self.mayavi_widget1.visualization.file_name = filename
        self.mayavi_widget1.visualization.update_plot()

    def load_selected_xrays2(self):
        filename = os.path.join(self.menu2.wd_info.text(),self.menu2.combobox_studyid.currentText())
        self.mayavi_widget2.visualization.file_name = filename
        self.mayavi_widget2.visualization.update_plot()


    def showSTL1(self):
        #todo: load from pickle directly
        if self.currentSTL is not None:
            self.viewer1.removeItem(self.currentSTL)
            self.viewer1.clear()
        file_name = os.path.join(self.menu1.wd_info.text(),self.menu1.combobox_studyid.currentText())
        with open(file_name,'rb') as fp:
            data = pickle.load(fp)
        for key in ['RPel']:
            f_name = data['surface'][key]['mesh_loc']
            points = data['surface'][key]['points']
            faces  = data['surface'][key]['points']
            #points,faces = self.loadSTL(f_name)


            mean_pos = np.mean(points,axis=0)
            #points = points-mean_pos
            self.viewer1.pan(mean_pos[0],mean_pos[1],mean_pos[2],relative='global')
            #self.viewer1.updateGL()
            print('mean osition of object is')
            print(mean_pos)
            print('camera position is')
            print(self.viewer1.cameraPosition())

        meshdata = gl.MeshData(vertexes=points,faces=faces)
        mesh = gl.GLMeshItem(meshdata=meshdata,smooth=True,drawFaces=True,drawEdges=False,edgeColor=(0,1,0,1),
                             shader='shaded')
        self.viewer1.addItem(mesh)
        self.viewer1.update()
        self.currentSTL = mesh


    def showSTL2(self):
        #todo: load from pickle directly
        if self.currentSTL is not None:
            self.viewer2.removeItem(self.currentSTL)
            self.viewer2.clear()
        file_name = os.path.join(self.menu2.wd_info.text(),self.menu2.combobox_studyid.currentText())
        with open(file_name,'rb') as fp:
            data = pickle.load(fp)
        for key in ['RPel']:
            f_name = data['surface'][key]['mesh_loc']
            # points = data['surface'][key]['points']
            # faces  = data['surface'][key]['points']
            points,faces = self.loadSTL(f_name)


            mean_pos = np.mean(points,axis=0)
            #points = points-mean_pos
            self.viewer2.pan(mean_pos[0],mean_pos[1],mean_pos[2],relative='global')
            self.viewer2.updateGL()
            print('mean osition of object is')
            print(mean_pos)
            print('camera position is')
            print(self.viewer2.cameraPosition())

            meshdata = gl.MeshData(vertexes=points,faces=faces)
            mesh = gl.GLMeshItem(meshdata=meshdata,smooth=True,drawFaces=True,drawEdges=False,edgeColor=(0,1,0,1),
                             shader='shaded')
        self.viewer2.addItem(mesh)
        self.viewer2.update()
        self.currentSTL = mesh

    def loadSTL(self,filename):
        m = mesh.Mesh.from_file(filename)
        shape = m.points.shape
        points = m.points.reshape(-1,3).astype(int)
        print(points.shape)
        faces = np.arange(points.shape[0]).reshape(-1,3).astype(int)
        return points,faces




if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())