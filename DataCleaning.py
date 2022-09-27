import os.path

from read_landmarks import *
from datetime import datetime
import pickle
from mayavi import mlab
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

source_loc = './data/Segmentation_and_landmarks_raw/'
target_loc = './data/Segmentation_and_landmarks_processed/'

paths     = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']



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


                        # point_cloud             = mesh.Mesh.from_file(stl_path)
                        if stl_path is not None:
                            surface_dict[key]       = {}
                            #todo: should I store the individual v0-v2?
                            #todo: find a way to use this data to plot a triangulated mesh instead of using the file
                            # name directly
                            # surface_dict[key]['v0'] = point_cloud.v0
                            # surface_dict[key]['v1'] = point_cloud.v1
                            # surface_dict[key]['v2'] = point_cloud.v2
                            # surface_dict[key]['x'] = point_cloud.x
                            # surface_dict[key]['y'] = point_cloud.y
                            # surface_dict[key]['z'] = point_cloud.z
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
    _to_pickle()









