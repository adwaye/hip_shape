from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont

import os
import numpy as np
from numpy import cos
from mayavi import mlab
from visualiser_utils import xray_selection_menu


os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui,QtCore
from traits.api import HasTraits,Instance,on_trait_change
from traitsui.api import View,Item
from mayavi.core.ui.api import MayaviScene,MlabSceneModel,SceneEditor
import pickle


## create Mayavi Widget and show

class Visualization(HasTraits):
    # file_name = "/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_raw/TOH - " \
    #             "Controls/C4/Left/C4_LPEL.stl"
    file_name = "/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - DDH/D11.p"

    scene = Instance(MlabSceneModel,())

    @on_trait_change('scene.activated')
    def update_plot(self):
        mlab.clf()
        ## PLot to Show
        with open(self.file_name,'rb') as fp:
            data = pickle.load(fp)
        for key in ['RPel','LPel']:
            f_name = data['surface'][key]['mesh_loc']
            m_data = mlab.pipeline.open(f_name)
            obj = mlab.pipeline.surface(m_data)

        # x = my_mesh.x
        # y = my_mesh.y
        # z = my_mesh.z
        # mlab.points3d(x, y, z)
        # mlab.show()


        color = [(139 / 255,233 / 255,253 / 255),(80 / 255,250 / 255,123 / 255),(139 / 255,233 / 255,253 / 255),
                 (80 / 255,250 / 255,123 / 255)]
        i = 0
        for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
            try:
                coords = data['landmarks'][key]
                mlab.plot3d(coords[:,0],coords[:,1],coords[:,2],line_width=10,color=color[i])
                mlab.points3d(coords[:,0],coords[:,1],coords[:,2],scale_factor=1,color=color[i])
            except KeyError:
                print("could not find "+key)
            i += 1


    view = View(Item('scene',editor=SceneEditor(scene_class=MayaviScene),
                     height=250,width=300,show_label=False),
                resizable=True)


class MayaviQWidget(QtGui.QWidget):
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self,parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)







#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def __init__(self):
        self.output_loc = './'


    def setupUi(self,MainWindow):
        ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200,200,1100,700)

        ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        ## GRID LAYOUT
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        ## BUTTONS
        self.menu1 = xray_selection_menu()
        scrollbar1 = QScrollArea(widgetResizable=True)
        scrollbar1.setMaximumHeight(300)
        scrollbar1.setWidget(self.menu1)
        self.gridLayout.addWidget(scrollbar1,0,0,1,1)
        #self.gridLayout.addWidget(self.qline_edit1,0,0,2,1)

        self.menu2 = xray_selection_menu()
        scrollbar2 = QScrollArea(widgetResizable=True)
        scrollbar2.setMaximumHeight(300)
        scrollbar2.setWidget(self.menu2)
        self.gridLayout.addWidget(scrollbar2,1,1,1,1)
        ## Mayavi Widget 1
        container1 = QtGui.QWidget()
        self.mayavi_widget1 = MayaviQWidget(container1)
        self.gridLayout.addWidget(self.mayavi_widget1,1,0,1,1)
        ## Mayavi Widget 2
        container2 = QtGui.QWidget()
        self.mayavi_widget2 = MayaviQWidget(container2)
        self.gridLayout.addWidget(self.mayavi_widget2,0,1,1,1)

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
        self.menu1.combobox_xrayid.clear()
        self.menu1.combobox_studyid.clear()
        self.display_studies1()

    def change_wd2(self):
        self.output_loc = self.menu2.wd_info.text()
        #self.image_widget.output_loc = self.output_loc
        if not os.path.isdir(self.output_loc):
            os.makedirs(self.output_loc)
        # self.xray_selection_menu.wd_info.setText(self.output_loc)
        self.menu2.combobox_xrayid.clear()
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
        self.menu1.combobox_studyid.currentIndexChanged.connect(self.display_CT1)
        self.menu2.combobox_studyid.currentIndexChanged.connect(self.display_CT2)
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




if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())