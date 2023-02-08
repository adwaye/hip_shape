"""
Contains some test methods on mesh resamplig with meshlab
"""

import pymeshlab as ml
import numpy
from data_utils import HipData,points2mesh3d,stl2mesh3d
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import meshlib.mrmeshpy as mr
import stl



def resample_mesh(hip: HipData)->HipData:

    pass


def resample_mesh_meshlab():
    """Test method to resample meshes.

    :return:
    :rtype:
    """
    fpath = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - Controls/C18.p'

    hip_data = HipData(fpath)

    m = ml.Mesh(*hip_data.RPEL)

    ms = ml.MeshSet()
    #ms.add_mesh(mesh=m,mesh_name='original Mesh')
    ms.load_new_mesh(hip_data.data['surface']['RPel']['mesh_loc'])



    #Target number of vertex
    TARGET = 9999

    #Estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2 * TARGET

    #Simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > TARGET):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation',targetfacenum=numFaces,preservenormal=True)
        print("Decimated to",numFaces,"faces mesh has",ms.current_mesh().vertex_number(),"vertex")
        #Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

    m = ms.current_mesh()

    points = m.vertex_matrix()
    faces  = m.face_matrix()

    vertices,I,J,K = points2mesh3d(points)  #-np.mean(self.RPEL[0],axis=0,keepdims=True))
    x,y,z = vertices.T
    right_mesh_plot = go.Mesh3d(x=x
                                ,y=y
                                ,z=z
                                ,i=I
                                ,j=J
                                ,k=K
                                ,color='blue',opacity=0.50)

    fig =  go.Figure(data=[right_mesh_plot
                          ])

    fig.show()

if __name__=='__main__':
    fpath = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - Controls/C18.p'

    hip_data = HipData(fpath)

    mesh = mr.loadMesh(mr.Path(hip_data.data['surface']['RPel']['mesh_loc']))

    # decimate it with max possible deviation 0.5:
    settings = mr.DecimateSettings()
    settings.maxError = 0.5
    settings.maxEdgeLen = 2.5
    result = mr.decimateMesh(mesh.value(),settings)
    print(result.facesDeleted)
    # 708298
    print(result.vertsDeleted)
    # save low-resolution mesh:
    filename = 'simplified-busto.stl'
    mr.saveMesh(mesh.value(),mr.Path('simplified-busto.stl'))
    m = stl.mesh.Mesh.from_file(filename)
    vertices,I,J,K = stl2mesh3d(m)
    x,y,z = vertices.T
    right_mesh_plot = go.Mesh3d(x=x
                                ,y=y
                                ,z=z
                                ,i=I
                                ,j=J
                                ,k=K
                                ,color='blue',opacity=0.50)

    fig =  go.Figure(data=[right_mesh_plot
                          ])
    fig.show()