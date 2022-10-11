import os.path

from read_landmarks import *
from datetime import datetime
import pickle
from mayavi import mlab
import numpy as np
import numpy.linalg

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("-f", "--file", dest="filename",
#                     help="write report to FILE", metavar="FILE")
# parser.add_argument("-q", "--quiet",
#                     action="store_false", dest="verbose", default=True,
#                     help="don't print status messages to stdout")
#
# args = parser.parse_args()


dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")

source_loc           = './data/Segmentation_and_landmarks_raw/'
target_loc           = './data/Segmentation_and_landmarks_processed/'
APP_aligned_loc      = './data/Segmentation_and_landmarks_APP_aligned/'
Socket_aligned_trans = './data/Segmentation_and_landmarks_socket_aligned/'

paths     = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']


# Relevant links:
#   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
#   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)

# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.
def umeyama(P,Q):
    assert P.shape == Q.shape
    n,dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP),centeredQ) / n

    V,S,W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    R = np.dot(V,W)

    varP = np.var(P,axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

    return c,R,t


def rigid_align(P,Q):
    #todo: the landmarks for the socket plane do not have the same number of landmarks. use distance transform method
    # for now
    #domain = 3d array containing both sets of points, set it to 1 where the curve lives

    #dist = distance transform
    #below are variables to go into scipy minimiser
    #A = rot
    #c = scale
    #t = trans
    return umeyama(P,Q)

def _to_pickle():
    log_loc  = './log'
    if not os.path.isdir(log_loc):os.makedirs(log_loc)
    log_file = os.path.join(log_loc,'stl_file_outside_sub_'+dt_string+'.txt')

    with open(log_file,'w') as f:
        f.write('')
    missing_file = os.path.join(log_loc,'missing_data' + dt_string + '.txt')
    with open(missing_file,'w') as g:
        g.write(' ')

    for p in paths:
        out_dict     = {}
        surface_dict = {}
        landmark_dict = {}
        source_path = os.path.join(source_loc,p)
        target_path = os.path.join(target_loc,p)

        if not os.path.isdir(target_path): os.makedirs(target_path)

        studies = [f for f in sorted(os.listdir(source_path)) if os.path.isdir(os.path.join(source_path,f))]
        for s in studies:

            source_study_path = os.path.join(source_path,s)
            print("========================================================================================")
            print("Extracting socket landmarks for " + source_study_path)
            xlsx_file_list    = sorted([f for f in os.listdir(source_study_path) if f.split('.')[-1] == "xlsx"])
            if len(xlsx_file_list)==0:
                temp_list = []
                temp_folders = [f for f in os.listdir(source_study_path) if os.path.isdir(os.path.join(
                    source_study_path,f))]
                kk = 0
                while len(temp_list)==0:
                    temp_list = sorted([f for f in os.listdir(os.path.join(
                    source_study_path,temp_folders[kk])) if f.split('.')[-1] == "xlsx"])
                    kk +=1
                if len(temp_list)>0:xlsx_file_list=temp_list.copy()
                xlsx_file         = xlsx_file_list[0]
                xlsx_path         = os.path.join(os.path.join(
                    source_study_path,temp_folders[kk]),xlsx_file)
            else:
                xlsx_file         = xlsx_file_list[0]
                xlsx_path         = os.path.join(source_study_path,xlsx_file)
            workbook          = openpyxl.load_workbook(xlsx_path,data_only=True)
            worksheet         = workbook.active

            origin,spacing = find_imOriginSpacing_from_worksheet(worksheet)
            #-----------------------------------------------------------------------------------------------------------
            #----if there is no spacing or origin information, data point is not useful hence we shouod not store it-
            #-----------------------------------------------------------------------------------------------------------
            if spacing is None or origin is None:
                with open(missing_file,'a') as g:
                    g.write(source_study_path+' \n')
                print("origin or spacing missing")
                continue
            else:

                if source_study_path.split('/')[-1] == 'UCLH - Dysplastics':
                    spacing[:,[2,1]] = spacing[:,[1,2]]
                    spacing = spacing * np.array([[1,1,-1]])
                my_dict = find_structure_coordinate_socket(worksheet)
                for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
                    coords = find_coordinates_from_worksheet(worksheet=worksheet,my_dict=my_dict,key=key)
                    print(key)
                    #print(coords)
                    if coords is not None:
                        print('spacing')
                        print(spacing==None)
                        print('origin')
                        print(origin == None)
                        coords = coords * spacing + (origin)

                        landmark_dict[key] = coords
                print("Extracting APP landmarks for " + source_study_path)
                temp_dict = find_app_coordinates(worksheet)
                for key,val in temp_dict.items():
                    landmark_dict[key] = np.array(val)

                #---------------------------------------------------------------------------------------------------------------
                #----Extracting landmark data-----------------------------------------------------------------------------------

                #---------------------------------------------------------------------------------------------------------------
                #----Extracting the surface data--------------------------------------------------------------------------------
                for side,key in zip(['right','left'],['RPel','LPel']):
                    folders = [f for f in os.listdir(source_study_path) if
                               os.path.isdir(os.path.join(source_study_path,f)) and side in f.lower()]
                    if len(folders) > 0:
                        study_subpath = os.path.join(source_study_path,folders[0])
                        print('--------------------------------------------------------------------------')
                        print('Processing '+study_subpath)
                        stl_file_list = [f for f in os.listdir(study_subpath) if f.split('.')[-1] == 'stl']
                        if len(stl_file_list) > 0:
                            stl_file = stl_file_list[0]
                            stl_path = os.path.join(study_subpath,stl_file)
                        else:#check that the stl file is not outisde of the subpath knstead of being in
                            # filepath/right, it is in filepath
                            stl_file_list = [f for f in os.listdir(source_study_path) if f.split('.')[-1] == 'stl']
                            if len(stl_file_list)>0:
                                stl_path = os.path.join(source_study_path,stl_file_list[0])
                            else:
                                stl_path = None
                            with open(log_file,'w') as f:
                                f.write(source_study_path+' \n')



                        if stl_path is not None:
                            m = mesh.Mesh.from_file(stl_path)
                            points,faces = loadSTL(stl_path)
                            surface_dict[key]       = {}
                            surface_dict[key]['points'] = points
                            surface_dict[key]['faces']  = faces
                            surface_dict[key]['mesh_loc'] = stl_path
                    else:
                        continue
                if stl_path is not None:
                    out_dict['surface']     = surface_dict
                    out_dict['landmarks']   = landmark_dict
                    print('Finished ' + study_subpath)
                            #with open('')
                    target_file = os.path.join(target_path,s+'.p')
                    with open(target_file,'wb') as fp:
                        pickle.dump(out_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)


def _create_aligned_data_APP():
    pass



def loadSTL(filename):
    m = mesh.Mesh.from_file(filename)
    shape = m.points.shape
    points = m.points.reshape(-1,3)
    print(points.shape)
    faces = np.arange(points.shape[0]).reshape(-1,3)
    return points,faces

def plot_all():
    target_file = "/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/UCLH - Controls/00796671.p"
    target_file = "/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_processed/TOH - DDH/D11.p"
    with open(target_file, 'rb') as fp:
        data = pickle.load(fp)

    # x = data['surface']['RPel']['x']
    # y = data['surface']['RPel']['y']
    # z = data['surface']['RPel']['z']
    # #mlab.mesh(x[0,:],x[1,:],x[2,:])
    # # mlab.pipeline.surface(mlab.pipeline.grid_source(x,y,z))
    # mlab.points3d(x,y,z)#
    for key in ['RPel','LPel']:
        file_name = data['surface'][key]['mesh_loc']

        # x = my_mesh.x
        # y = my_mesh.y
        # z = my_mesh.z
        # mlab.points3d(x, y, z)
        # mlab.show()
        m_data = mlab.pipeline.open(file_name)
        s = mlab.pipeline.surface(m_data)
    color = [(139/255, 233/255, 253/255),(80/255, 250/255, 123/255),(139/255, 233/255, 253/255),(80/255, 250/255, 123/255)]
    i =0
    for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
        coords = data['landmarks'][key]
        mlab.plot3d(coords[:,0],coords[:,1],coords[:,2],line_width=10,color=color[i])
        mlab.points3d(coords[:,0],coords[:,1],coords[:,2],scale_factor=1,color=color[i])
        i +=1

    mlab.show()

if __name__=='__main__':
    #_to_pickle()
    location       = os.path.join(target_loc,paths[0])
    files          = [os.path.join(location,f) for f in os.listdir(location)]
    template_file  = files[0]
    target_file    = files[2]

    with open(template_file,'rb') as fp:
        template_data = pickle.load(fp)

    with open(target_file,'rb') as fp:
        target_data = pickle.load(fp)


    k = 0
    for key in ['RASIS','LASIS','RTUB','LTUB']:

        try:
            if k ==0:
                template_points = np.expand_dims(template_data['landmarks'][key],axis=0)
                target_points   = np.expand_dims(target_data['landmarks'][key],axis=0)
            else:
                template_points_ = np.expand_dims(template_data['landmarks'][key],axis=0)
                template_points  = np.concatenate((template_points_,template_points),axis=0)
                target_points_   = np.expand_dims(target_data['landmarks'][key],axis=0)
                target_points    = np.concatenate((target_points_,target_points),axis=0)

            k+=1
        except KeyError:
            print('cannot find key '+key)

    c,R,t = umeyama(target_points,template_points)

    err = ((template_points.dot(c * R) + t - target_points) ** 2).sum()
    transformed_points = target_points.dot(c * R) + t
    print('error is {:}',format(err))

    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(template_points[:,0],template_points[:,1],template_points[:,2],color='green')
    ax.scatter(transformed_points[:,0],transformed_points[:,1],transformed_points[:,2],'.',color='red')
    ax.scatter(target_points[:,0],target_points[:,1],target_points[:,2],color='blue')
    ax.set_box_aspect(aspect=(1,1,1))


    app_alignment_dict = {}
    app_alignment_dict['R'] = R
    app_alignment_dict['c'] = c
    app_alignment_dict['t'] = t
    #todo: once I find a way to sample the socket lines to the same number of points, I can rigid align the socket
    # points as well
    socket_alignment_dict = {}
    for side in ['Right','Left']:
        k = 0
        for key in ['Ant Lat','Post Lat']:
            try:
                if k == 0:
                    template_points = template_data['landmarks'][side+' '+key]
                    target_points   = target_data['landmarks'][side+' '+key]
                else:
                    template_points_ = template_data['landmarks'][side+' '+key]
                    template_points  = np.concatenate((template_points_,template_points),axis=0)
                    target_points_   = target_data['landmarks'][side+' '+key]
                    target_points    = np.concatenate((target_points_,target_points),axis=0)

                k += 1
            except KeyError:
                print('cannot find key ' + key)
        fig = plt.figure(figsize=(4,4))

        ax = fig.add_subplot(111,projection='3d')
        fig.suptitle('haha')

        ax.scatter(template_points[:,0],template_points[:,1],template_points[:,2],color='green')
        #ax.scatter(transformed_points[:,0],transformed_points[:,1],transformed_points[:,2],'.',color='red')
        ax.scatter(target_points[:,0],target_points[:,1],target_points[:,2],color='blue')
        ax.set_box_aspect(aspect=(1,1,1))
        #c,R,t = rigid_align(target_points,template_points)
        # socket_alignment_dict[side] = {}
        # socket_alignment_dict[side]['R'] = R
        # socket_alignment_dict[side]['c'] = c
        # socket_alignment_dict[side]['t'] = t


























