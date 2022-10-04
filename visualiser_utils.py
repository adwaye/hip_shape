from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont

import os
import numpy as np



def stl2mesh3d(stl_mesh):
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

class xray_selection_menu(QWidget):
    """
    class that handles the xray creation and the damage type selector
    """
    def __init__(self):
        super(xray_selection_menu,self).__init__()
        self.font_header = QFont('Android Roboto', 15)
        self.font_subheader = QFont('Android Roboto', 13)
        self.font_text = QFont('Android Roboto', 10)
        self.font_button = QFont('Android Roboto', 11)
        self.temp_name   = None

        self.layout = QVBoxLayout()
        self.init_xray_creation_options()

        self.init_damage_selector()
        self.setLayout(self.layout)
        self.connect_buttons()

    def init_xray_creation_options(self):
        layout = QVBoxLayout()
        study_layout = QHBoxLayout()
        study_id_label = QLabel("Study ID")
        study_id_label.setFont(self.font_text)
        study_layout.addWidget(study_id_label,1)  # Number for relative size compared to other widgets
        self.combobox_studyid = QComboBox()
        self.combobox_studyid.setFont(self.font_text)
        self.combobox_studyid.setToolTip(
            "This displays the file study id to which the xray being viewed belongs. Use the dropdown to select another study from the working directory.")
        study_layout.addWidget(self.combobox_studyid,2)  # Number for relative size compared to other widgets
        studyWidget = QWidget()
        studyWidget.setLayout(study_layout)

        xray_layout = QHBoxLayout()
        xray_id_label = QLabel("Xray ID")
        xray_id_label.setFont(self.font_text)
        xray_layout.addWidget(xray_id_label,1)  # Number for relative size compared to other widgets
        self.combobox_xrayid = QComboBox()
        self.combobox_xrayid.setToolTip("This displays the file name of the xray being viewed. Use the dropdown to select another xray from the same study.")
        self.combobox_xrayid.setFont(self.font_text)
        xray_layout.addWidget(self.combobox_xrayid,2)  # Number for relative size compared to other widgets
        xray_widget = QWidget()
        xray_widget.setLayout(xray_layout)
        layout.addWidget(studyWidget)
        layout.addWidget(xray_widget)

        button_layout = QHBoxLayout()
        self.new_study_button = QPushButton("+ New Study")
        self.new_study_button.setStyleSheet(
            "QPushButton" "{" "background-color: rgb(102,102,102); color: white" "}" "QPushButton::pressed" "{"
            "background-color: #362699; color: white" "}")

        self.addXrayToStudy_button = QPushButton("Add X-ray to Study")
        self.addXrayToStudy_button.setToolTip("Opens up the file dialog to allow the the user to select a new xray to add to the current study.")
        self.addXrayToStudy_button.setStyleSheet(
            "QPushButton" "{" "background-color: rgb(102,102,102); color: white" "}" "QPushButton::pressed" "{"
            "background-color: #362699; color: white" "}")
        button_layout.addWidget(self.new_study_button)
        button_layout.addWidget(self.addXrayToStudy_button)

        self.set_wdir_button = QPushButton("Change save location")
        self.set_wdir_button.setToolTip("Changes the location where xrays and annotations are saved. If a location with a supported file structure is found, the app loads all the studies for the user.")
        self.set_wdir_button.setStyleSheet(
            "QPushButton" "{" "background-color: rgb(102,102,102); color: white" "}" "QPushButton::pressed" "{"
            "background-color: #362699; color: white" "}")
        button_layout.addWidget(self.set_wdir_button)

        layout3 = QHBoxLayout()
        wd_label         = QLabel("Working Directory")

        self.wd_info  = QLineEdit()
        self.wd_info.setReadOnly(True)

        layout3.addWidget(wd_label, 1)
        layout3.addWidget(self.wd_info, 2)


        layout4 = QHBoxLayout()
        file_label         = QLabel("Current file")

        self.current_file_info  = QLineEdit()
        self.current_file_info.setReadOnly(True)

        self.current_study_info = QLineEdit()
        self.current_study_info.setReadOnly(True)

        layout4.addWidget(file_label, 1)
        # layout4.addWidget(self.current_file_info, 2)
        # layout4.addWidget(self.c)



        layout2 = QHBoxLayout()
        score_date_label         = QLabel("Date")
        score_organ_label        = QLabel("Anatomical Structure")
        self.xray_info_box_date  = QLineEdit()
        self.xray_info_box_date.setReadOnly(True)
        self.xray_info_box_date.setToolTip("Shows the date of acquisition of the xray being viewed")
        self.xray_info_box_organ = QLineEdit()

        self.xray_info_box_organ.setReadOnly(True)
        self.xray_info_box_organ.setToolTip("Shows the name of the organ being displayed")
        layout2.addWidget(score_date_label, 1)
        layout2.addWidget(self.xray_info_box_date, 2)
        layout2.addWidget(score_organ_label, 1)
        layout2.addWidget(self.xray_info_box_organ, 2)



        widget_wd = QWidget()
        widget_wd.setLayout(layout3)
        widget_options = QWidget()
        widget_options.setLayout(layout)
        widget_info    = QWidget()
        widget_info.setLayout(layout2)
        widget_buttons = QWidget()
        widget_buttons.setLayout(button_layout)
        widget_file = QWidget()
        widget_file.setLayout(layout4)
        self.layout.addWidget(widget_wd)
        # self.layout.addWidget(widget_file)
        self.layout.addWidget(widget_options)
        self.layout.addWidget(widget_info)
        self.layout.addWidget(widget_buttons)



    def init_damage_selector(self):
        layout = QHBoxLayout()
        score_id_label = QLabel("Scoring Method")
        score_id_label.setFont(self.font_text)
        layout.addWidget(score_id_label,1)

        self.score_selector = QComboBox()
        self.score_selector.setFont(self.font_text)
        # self.score_selector.setStyleSheet(
        #     "QPushButton" "{" "background-color: rgb(102,102,102); color: white" "}" "QPushButton::pressed" "{"
        #     "background-color: #362699; color: white" "}")
        layout.addWidget(self.score_selector,2)
        widget = QWidget()
        widget.setLayout(layout)
        self.layout.addWidget(widget)


    def getDirectory(self):
        response = QFileDialog.getExistingDirectory(
            self,
            caption='Select a folder',directory=self.wd_info.text()

        )

        return response


    def getFiles(self):
        response = QFileDialog.getOpenFileName(
            self,
            caption='Select a file'
        )
        self.temp_name = response[0]
        print("file fed to xrayselectionmenu is")
        print(response)
        return response


    def change_wd(self):
        response = self.getDirectory()
        if response != '':
            self.wd_info.setText(os.path.join(os.sep,response))

    def add_xray_to_study(self):
        file = self.getFiles()
        if file[0] != '':
            self.current_file_info.setText(os.path.join(os.sep,file[0]))

    def create_new_study(self):
        file = self.getFiles()
        if file[0] != '':
            self.current_study_info.setText(os.path.join(os.sep,file[0]))

    def connect_buttons(self):
        self.set_wdir_button.clicked.connect(self.change_wd)
        self.addXrayToStudy_button.clicked.connect(self.add_xray_to_study)
        self.new_study_button.clicked.connect(self.create_new_study)
