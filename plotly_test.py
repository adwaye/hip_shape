import plotly
plotly.__version__
import numpy as np
from stl import mesh  # pip install numpy-stl
import plotly.graph_objects as go


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


my_mesh = mesh.Mesh.from_file('data/Segmentation_and_landmarks_raw/TOH - FAI/F1/Left/F1_LPEL.stl')
# stl file from: https://github.com/stephenyeargin/stl-files/blob/master/AT%26T%20Building.stl
my_mesh.vectors.shape

vertices, I, J, K = stl2mesh3d(my_mesh)
x, y, z = vertices.T

colorscale= [[0, '#e5dee5'], [1, '#e5dee5']]
mesh3D = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=I,
            j=J,
            k=K,
            flatshading=True,
            colorscale=colorscale,
            intensity=z,
            name='AT&T',
            showscale=False)




title = "Mesh3d from a STL file<br>AT&T building"
layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
            title_text=title, title_x=0.5,
                   font_color='white',
            width=800,
            height=800,
            scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
            scene_xaxis_visible=False,
            scene_yaxis_visible=False,
            scene_zaxis_visible=False)


fig = go.Figure(data=[mesh3D], layout=layout)


fig.data[0].update(lighting=dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2,facenormalsepsilon=0))
fig.data[0].update(lightposition=dict(x=300,
                                      y=300,
                                      z=1000))

fig.show()
# import chart_studio.plotly as py
# py.iplot(fig, filename='ATandT-building')



# from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
# import plotly.express as px
#
#
# class Widget(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.button = QtWidgets.QPushButton('Plot', self)
#         self.browser = QtWebEngineWidgets.QWebEngineView(self)
#
#         vlayout = QtWidgets.QVBoxLayout(self)
#         vlayout.addWidget(self.button, alignment=QtCore.Qt.AlignHCenter)
#         vlayout.addWidget(self.browser)
#
#         self.button.clicked.connect(self.show_graph)
#         self.resize(1000,800)
#
#     def show_graph(self):
#         df = px.data.tips()
#         fig = px.box(df, x="day", y="total_bill", color="smoker")
#         fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
#         self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#     widget = Widget()
#     widget.show()
#     app.exec()