from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
from tensorboard.plugins.mesh import summary as mesh_summary
import numpy as np
import os
import trimesh

from medpy.io import load
import matplotlib.pyplot as plt
from read_landmarks import *




def mesh_tf_demo():
    sample_mesh = 'https://storage.googleapis.com/tensorflow-graphics/tensorboard/test_data/ShortDance07_a175_00001.ply'

    # Read all sample PLY files.
    mesh = trimesh.load_remote(sample_mesh)
    vertices = np.array(mesh.vertices)
    # Currently only supports RGB colors.
    colors = np.array(mesh.visual.vertex_colors[:,:3])
    faces = np.array(mesh.faces)

    # Add batch dimension, so our data will be of shape BxNxC.
    vertices = np.expand_dims(vertices,0)
    colors = np.expand_dims(colors,0)
    faces = np.expand_dims(faces,0)
    log_dir = './view'
    vertices_tensor = tf.placeholder(tf.float32,vertices.shape)
    faces_tensor = tf.placeholder(tf.int32,faces.shape)
    colors_tensor = tf.placeholder(tf.int32,colors.shape)

    meshes_summary = mesh_summary.op(
        'mesh_color_tensor',vertices=vertices_tensor,faces=faces_tensor,
        colors=colors_tensor)

    # Create summary writer and session.
    writer = tf.summary.FileWriter(log_dir)
    with tf.Session() as sess:
        summaries = sess.run([meshes_summary],feed_dict={
            vertices_tensor:vertices,
            faces_tensor:faces,
            colors_tensor:colors,
        })
        # Save summaries.
        for summary in summaries:
            writer.add_summary(summary)


def show_3d_slicing():
    data_loc = './data/Andrew_Richie_pelvis_shape/'

    # image_data, image_header = load(os.path.join(data_loc,'Labels_segments.mha'))
    image_data, image_header = load(os.path.join(data_loc,'wholepelvis_stack.mha'))

    Dpth = image_data.shape[-1]
    fig,ax = plt.subplots()
    for d in range(Dpth):
        ax.clear()
        ax.imshow(image_data[:,:,d])
        ax.set_title('Slice {:}'.format(d))
        plt.pause(0.01)




# Create a new plot

# point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
def tf_make_tb_mesh(data_loc = './data/Andrew_Richie_pelvis_shape'):
    """
    reads all stl files in a folder and concatantes them into a point cloud that is then saved as a tensorboard object

    :param data_loc: location of the .stl files
    :type data_loc:  string
    :return:
    :rtype:
    """

    # pelvis_mesh = mesh.Mesh.from_file(os.path.join(data_loc,'pelvis.stl'))
    # rf_mesh     = mesh.Mesh.from_file(os.path.join(data_loc,'RightFemur.stl'))
    # lf_mesh     = mesh.Mesh.from_file(os.path.join(data_loc,'LeftFemur.stl'))

    files = [os.path.join(data_loc,f) for f in os.listdir(data_loc) if f.split('.')[-1]=='stl']
    if len(files)!=0:

        k = 0
        for f in files:
            if k ==0:
                point_cloud = mesh.Mesh.from_file(f).v0
            else:
                point_cloud = np.concatenate((point_cloud,mesh.Mesh.from_file(f).v0),axis=0)
            k+=1
    # point_cloud = np.concatenate((pelvis_mesh.v0,rf_mesh.v0,lf_mesh.v0),axis=0)

        N = point_cloud.shape[0]
        D = point_cloud.shape[1]
        point_cloud=point_cloud.reshape((1,N,D))

    # point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

    # summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)


    # # Add batch dimension, so our data will be of shape BxNxC.
    # vertices = np.expand_dims(vertices, 0)
    # colors = np.expand_dims(colors, 0)
    # faces = np.expand_dims(faces, 0)


        tf_point_cloud = tf.placeholder(tf.float32, point_cloud.shape)
        # vertices_tensor = tf.placeholder(tf.float32, vertices.shape)
        # faces_tensor = tf.placeholder(tf.int32, faces.shape)
        #colors_tensor = tf.placeholder(tf.int32, colors.shape)

        meshes_summary = mesh_summary.op(
            'mesh_color_tensor', vertices=tf_point_cloud)#,faces=tf_faces)

        # Create summary writer and session.
        sub_name   = os.path.basename(data_loc)

        study_name = data_loc.split('/')[-2]
        log_dir = os.path.join('./view',study_name)
        log_dir = os.path.join(log_dir,sub_name)
        print(log_dir)
        if not os.path.isdir(log_dir):  os.makedirs(log_dir)
        writer = tf.summary.FileWriter(log_dir)


        with tf.Session() as sess:
            summaries = sess.run([meshes_summary], feed_dict={
            tf_point_cloud: point_cloud
            })
        # Save summaries.
            for summary in summaries:
                writer.add_summary(summary)
    else:
        print('no stl files found in '+ data_loc)

def tf_make_tb_mesh_with_landmarks(data_loc = './data/Segmentation_and_landmarks_raw/UCLH - Controls/68567126'):
    """
    reads all stl files in a folder and concatantes them into a point cloud that is then saved as a tensorboard object

    :param data_loc: location of the .stl files
    :type data_loc:  string
    :return:
    :rtype:
    """

    # pelvis_mesh = mesh.Mesh.from_file(os.path.join(data_loc,'pelvis.stl'))
    # rf_mesh     = mesh.Mesh.from_file(os.path.join(data_loc,'RightFemur.stl'))
    # lf_mesh     = mesh.Mesh.from_file(os.path.join(data_loc,'LeftFemur.stl'))
    files = []
    for subfolder in ['Left','Right']:
        loc = os.path.join(data_loc,subfolder)

        files += [os.path.join(loc,f) for f in os.listdir(loc) if f.split('.')[-1]=='stl']

    if len(files)!=0:

        color_pal = [[255,184,108],[189,147,249],[40, 42, 54]]

        k = 0
        for f in files:
            if k ==0:
                point_cloud = mesh.Mesh.from_file(f).v0
                color = np.repeat([color_pal[k]],repeats=point_cloud.shape[0],axis=0)
            else:
                point_cloud = np.concatenate((point_cloud,mesh.Mesh.from_file(f).v0),axis=0)
                color = np.concatenate((color,np.repeat([color_pal[k]],repeats=point_cloud.shape[0],axis=0)),axis=0)

            k+=1
    # point_cloud = np.concatenate((pelvis_mesh.v0,rf_mesh.v0,lf_mesh.v0),axis=0)



        xlsx_file = [f for f in os.listdir(data_loc) if f.split('.')[-1] == "xlsx"][0]

        xlsx_path = os.path.join(data_loc,xlsx_file)

        workbook = openpyxl.load_workbook(xlsx_path,data_only=True)

        # Define variable to read the active sheet:
        worksheet = workbook.active

        my_dict = find_structure_coordinate_socket(worksheet)
        k = 0
        for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
            _coords = find_coordinates_from_worksheet(worksheet=worksheet,my_dict=my_dict,key=key)
            if k ==0:
                coords = _coords.copy()
            else:
                coords = np.concatenate((coords,_coords),axis=0)
            print(_coords)

        color = np.concatenate((color,np.repeat([color_pal[-1]],repeats=coords.shape[0],axis=0)),axis=0)
        point_cloud = np.concatenate((point_cloud,coords),axis=0)

        N = point_cloud.shape[0]
        D = point_cloud.shape[1]
        point_cloud = point_cloud.reshape((1,N,D))


    # # Add batch dimension, so our data will be of shape BxNxC.
    # vertices = np.expand_dims(vertices, 0)
    # colors = np.expand_dims(colors, 0)
    # faces = np.expand_dims(faces, 0)


        tf_point_cloud = tf.placeholder(tf.float32, point_cloud.shape)
        # vertices_tensor = tf.placeholder(tf.float32, vertices.shape)
        # faces_tensor = tf.placeholder(tf.int32, faces.shape)
        colors_tensor = tf.placeholder(tf.int32, color.shape)

        meshes_summary = mesh_summary.op(
            'mesh_color_tensor', vertices=tf_point_cloud)#,faces=tf_faces)

        # Create summary writer and session.
        sub_name   = os.path.basename(data_loc)

        study_name = data_loc.split('/')[-2]
        log_dir = os.path.join('./view',study_name)
        log_dir = os.path.join(log_dir,sub_name)
        print(log_dir)
        if not os.path.isdir(log_dir):  os.makedirs(log_dir)
        writer = tf.summary.FileWriter(log_dir)


        with tf.Session() as sess:
            summaries = sess.run([meshes_summary], feed_dict={
            tf_point_cloud: point_cloud,
            colors_tensor : color
            })
        # Save summaries.
            for summary in summaries:
                writer.add_summary(summary)
    else:
        print('no stl files found in '+ data_loc)


def tf_make_tb_mesh_left_right(data_loc = './data/Andrew_Richie_pelvis_shape',show_faces=True):
    """
    reads the study folders containng the right and left pelvis and stores them as a tensorboard mesh obejct
    :param data_loc: location of the right and left stl files: should contain folders with names Right and Left
    contaning one stl file for the right and left pelvis respectively
    :type data_loc: string
    :return:
    :rtype:
    """
    tf_color = tf.placeholder(tf.int32,shape=[1,None,3])
    tf_vertices = tf.placeholder(tf.float32,[1,None,3])
    tf_faces = tf.placeholder(tf.float32,[1,None,3])

    meshes_summary = mesh_summary.op(
        'mesh_color_tensor',vertices=tf_vertices,colors=tf_color)
        #faces=tf_faces)  #,faces=tf_faces)

    # Create summary writer and session.
    sub_name = os.path.basename(data_loc)

    study_name = data_loc.split('/')[-1]
    log_dir = os.path.join('./viewboth',study_name)
    #print(log_dir)
    if not os.path.isdir(log_dir):  os.makedirs(log_dir)
    writer = tf.summary.FileWriter(log_dir)

    rf_loc = os.path.join(data_loc,'Right')
    lf_loc = os.path.join(data_loc,'Left')
    locs   = [rf_loc,lf_loc]
    color_pal = [[255, 184, 108],[189, 147, 249]]

    files = [os.path.join(rf_loc,f) for f in os.listdir(rf_loc) if f.split('.')[-1]=='stl']
    with tf.Session() as sess:
        loc_index = 0
        for loc in locs :
            files = [os.path.join(loc,f) for f in os.listdir(loc) if f.split('.')[-1]=='stl']
            k=0
            for f in files:
                if loc_index == 0:
                    myobj = trimesh.load_mesh(f,enable_post_processing=True,solid=True)
                    #point_cloud = mesh.Mesh.from_file(f).v0
                    vertices = myobj.vertices
                    faces = myobj.faces
                    color = np.repeat([color_pal[loc_index]],repeats=vertices.shape[0],axis=0)
                else:
                    myobj = trimesh.load_mesh(f,enable_post_processing=True,solid=True)
                    vertices = np.concatenate((vertices,myobj.vertices),axis=0)
                    faces = np.concatenate((faces,myobj.faces),axis=0)
                    color = np.concatenate((color,np.repeat([color_pal[loc_index]],repeats=myobj.vertices.shape[0],axis=0)),
                                           axis=0)
                k+=1
            loc_index += 1

        N = vertices.shape[0]
        D = vertices.shape[1]
        vertices=vertices.reshape((1,N,D))
        color = color.reshape((1,N,D))
        N = faces.shape[0]
        D = faces.shape[1]
        faces = faces.reshape((1,N,D))


        summaries = sess.run([meshes_summary], feed_dict={
        tf_vertices: vertices,
        #tf_faces   : faces,
        tf_color   : color
        })
    # Save summaries.
        for summary in summaries:
            writer.add_summary(summary)



    #     k = 0
    #     for f in files:
    #         if k ==0:
    #             myobj = trimesh.load_mesh(f,enable_post_processing=True,solid=True)
    #             #point_cloud = mesh.Mesh.from_file(f).v0
    #             vertices = myobj.vertices
    #             faces    = myobj.faces
    #         else:
    #             myobj = trimesh.load_mesh(f,enable_post_processing=True,solid=True)
    #             vertices = np.concatenate((vertices,myobj.vertices),axis=0)
    #             faces = np.concatenate((faces,myobj.faces),axis=0)
    #         color = tf.repeat([[40,42,54]],repeats = vertices.shape[0],axis=0)
    #         k+=1
    #
    # files = [os.path.join(lf_loc,f) for f in os.listdir(lf_loc) if f.split('.')[-1] == 'stl']
    # if len(files) != 0:
    #
    #     k = 0
        # for f in files:
        #     myobj = trimesh.load_mesh(f,enable_post_processing=True,solid=True)
        #     vertices = np.concatenate((vertices,myobj.vertices),axis=0)
        #     faces = np.concatenate((faces,myobj.faces),axis=0)
        #     color_lf = tf.repeat([[255,85,85]],repeats = myobj.vertices.shape[0],axis=0)
        #
        #     k += 1
    # point_cloud = np.concatenate((pelvis_mesh.v0,rf_mesh.v0,lf_mesh.v0),axis=0)


        # tf_faces = tf.constant(faces)







if __name__=='__main__':
    #get stl from
    #show_3d_slicing()
    # data_loc = "./data/UCLH-Controls/97255839/Right"
    # os.listdir(data_loc)
    # tf_make_tb_mesh(data_loc=data_loc)
    # data_loc  = "./data/UCLH-Controls/97255839"
    # top_loc   = "./data/UCLH-Controls/"
    # data_locs = [os.path.join(top_loc,f) for f in os.listdir(top_loc) if os.path.isdir(os.path.join(top_loc,f))]
    # for data_loc in data_locs:
    #     tf_make_tb_mesh_left_right(data_loc=data_loc)
    # file = "data/UCLH-Controls/99782485/Right/99782485_RPEL.stl"
    # vertices = mesh.Mesh.from_file(file)
    # import trimesh
    #
    # myobj = trimesh.load_mesh(file, enable_post_processing=True, solid=True) # Import Objects
    #myobj.show()
    tf_make_tb_mesh_with_landmarks()
    #print(myobj.faces)