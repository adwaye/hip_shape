from stl import mesh
import os
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from mpl_toolkits.mplot3d import Axes3D

path = './data/UCLH-Controls'

studies = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

study = studies[0]


study_path = os.path.join(path,study)
xlsx_file = [f for f in os.listdir(study_path) if f.split('.')[-1]=="xlsx"][0]


xlsx_path  = os.path.join(study_path,xlsx_file)
# Define variable to load the wookbook
workbook = openpyxl.load_workbook(xlsx_path,data_only=True)

# Define variable to read the active sheet:
worksheet = workbook.active


# # Iterate the loop to read the cell values
# for i in range(0, worksheet.max_row):
#     for col in worksheet.iter_cols(1, worksheet.max_column):
#         print(col[i].value, end="\t\t")
#     print('')


def find_structure_coordinate_socket(worksheet):
    """
    Finds the coordinates of landmarks from an excel file. this assumes that the excel file does not follow any specific
    format and is arranged in a disorderly manner
    :param worksheet: object returned by openpyxl.load_workbook().active
    :return: dictionary of landmarks column and row where these coordinates point to where one should iterate to get
    the landmarks
    """


    # Iterate the loop to read the cell values
    #rows start at 1
    #columns start at 0
    my_dict = {}
    for i in range(0, worksheet.max_row):
        for col in worksheet.iter_cols(1, worksheet.max_column):
            if type(col[i].value) is str:
                if "ant lat" in col[i].value.lower():
                    print("found ant column at "+col[i].column_letter+str(col[i].row))

                    ant_col = col[i].column-1
                    ant_row = i+1
                    if type(worksheet[ant_row-1][ant_col].value) is str:
                        if "right" in worksheet[ant_row-1][ant_col].value.lower():
                            my_dict['Right Ant Lat Col'] = ant_col
                            my_dict['Right Ant Lat Row'] = ant_row
                    if type(worksheet[ant_row-1][ant_col].value) is str:
                        if "left" in worksheet[ant_row-1][ant_col].value.lower():
                            my_dict['Left Ant Lat Col'] = ant_col
                            my_dict['Left Ant Lat Row'] = ant_row
                elif "post lat" in col[i].value.lower():
                    print("found post column at "+col[i].column_letter+str(col[i].row))

                    ant_col = col[i].column-1
                    ant_row = i+1
                    if type(worksheet[ant_row-1][ant_col-3].value) is str:
                        if "right" in worksheet[ant_row-1][ant_col-3].value.lower():
                            print('saving right post lat coords')
                            my_dict['Right Post Lat Col'] = ant_col
                            my_dict['Right Post Lat Row'] = ant_row
                    if type(worksheet[ant_row-1][ant_col-3].value) is str:
                        if "left" in worksheet[ant_row-1][ant_col-3].value.lower():
                            my_dict['Left Post Lat Col'] = ant_col
                            my_dict['Left Post Lat Row'] = ant_row

    return my_dict






def find_coordinates_from_worksheet(worksheet,my_dict,key="Right Ant Lat",):
    """
    extracts the socket coordinates from an excelt sheet by using the coordinates where the search should start
    :param worksheet: loaded excel file by pyopenxl
    :param my_dict:   dictionary containing column and row indices for where the information can be found
    :param key: Which side and part of the socket to look for
                choices : "Right Ant Lat", "Right Post Lat", "Left Ant Lat", "Left Post Lat"
                these are keys of the dictionary output by find_structure_coordinate_socket
    :return:
    """
    row_val = my_dict[key+' Row']
    col_val = my_dict[key+' Col']
    ix=1
    jx=1

    while worksheet[row_val+ix][col_val].value is not None:
        coord_list = []
        for k in range(0,3):
            coord_list+=[worksheet[row_val+ix][col_val+k].value]
        if ix==1:
            coordinates = np.expand_dims(np.array(coord_list),0)
        else:
            coordinates = np.concatenate((coordinates,np.expand_dims(np.array(coord_list),0)),axis=0)
        ix+=1
    return coordinates


if __name__=='__main__':
    path = './data/Segmentation_and_landmarks_raw/UCLH - Controls'

    studies = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

    study = studies[2]




    study_path = os.path.join(path,study)
    study_subpath = os.path.join(study_path,'Left')
    stl_file   = [f for f in os.listdir(study_subpath) if f.split('.')[-1]=='stl'][0]
    xlsx_file = [f for f in os.listdir(study_path) if f.split('.')[-1] == "xlsx"][0]

    xlsx_path = os.path.join(study_path,xlsx_file)
    stl_path  = os.path.join(study_subpath,stl_file)
    # Define variable to load the wookbook
    workbook = openpyxl.load_workbook(xlsx_path,data_only=True)

    # Define variable to read the active sheet:
    worksheet = workbook.active

    my_dict = find_structure_coordinate_socket(worksheet)
    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111, projection='3d')

    print(study)
    for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
        coords  = find_coordinates_from_worksheet(worksheet=worksheet,my_dict=my_dict,key=key)
        print(key)
        print(coords)
        ax.scatter(coords[:,0],coords[:,1],coords[:,2])  # plot the point (2,3,4) on the figure

    point_cloud = mesh.Mesh.from_file(stl_path).v0
    ax.scatter(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2])
