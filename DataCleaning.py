from read_landmarks import *
from datetime import datetime
import h5py
import json,pickle


dt_string = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")

source_loc = './data/Segmentation_and_landmarks_raw/'
target_loc = './data/Segmentation_and_landmarks_processed/'

paths     = ['UCLH - Controls','TOH - Controls','TOH - FAI','TOH - DDH']

log_loc  = './log'
if not os.path.isdir(log_loc):os.makedirs(log_loc)
log_file = os.path.join(log_loc,'stl_file_outside_sub_'+dt_string+'.txt')
f = open(log_file,'w')

for p in paths[0:1]:
    out_dict     = {}
    surface_dict = {}
    landmark_dict = {}
    source_path = os.path.join(source_loc,p)
    target_path = os.path.join(target_loc,p)
    if not os.path.isdir(target_path): os.makedirs(target_path)
    studies = [f for f in sorted(os.listdir(source_path)) if os.path.isdir(os.path.join(source_path,f))]
    for s in studies[0:1]:
        source_study_path = os.path.join(source_path,s)
        xlsx_file         = [f for f in os.listdir(source_study_path) if f.split('.')[-1] == "xlsx"][0]
        xlsx_path         = os.path.join(source_study_path,xlsx_file)
        workbook          = openpyxl.load_workbook(xlsx_path,data_only=True)
        worksheet         = workbook.active
        print("========================================================================================")
        print("Extracting socket landmarks for "+source_study_path)
        origin,spacing = find_imOriginSpacing_from_worksheet(worksheet)

        if source_study_path.split('/')[-1] == 'UCLH - Dysplastics':
            spacing[:,[2,1]] = spacing[:,[1,2]]
            spacing = spacing * np.array([[1,1,-1]])
        my_dict = find_structure_coordinate_socket(worksheet)
        for key in ['Right Ant Lat','Right Post Lat','Left Ant Lat','Left Post Lat']:
            coords = find_coordinates_from_worksheet(worksheet=worksheet,my_dict=my_dict,key=key)
            print(key)
            print(coords)
            if coords is not None:
                mean_coords = np.mean(coords)
                #scaling = np.array([[0.8379,0.8379,1.5]])
                #origin  = np.array([[-209.1,-375.1,-760.5]])
                #coords = ((coords-mean_coords)*spacing)+mean_coords
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
                else:
                    stl_file_list = [f for f in os.listdir(source_study_path) if f.split('.')[-1] == 'stl']
                    stl_path = os.path.join(source_study_path,stl_file_list[0])
                    f.write(source_study_path)


                point_cloud             = mesh.Mesh.from_file(stl_path)
                surface_dict[key]       = {}
                surface_dict[key]['v0'] = point_cloud.v0
                surface_dict[key]['v1'] = point_cloud.v1
                surface_dict[key]['v2'] = point_cloud.v2
                out_dict['surface']     = surface_dict
                out_dict['landmarks']   = landmark_dict
                print('Finished ' + study_subpath)
                #with open('')
                with open('data.p','wb') as fp:
                    pickle.dump(out_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)
                # hf = h5py.File(os.path.join(target_path,s)+'.h5',"w")
                # hf.attrs.update(surface_dict)
                #hf.close()

f.close()


with open('data.p', 'rb') as fp:
    data = pickle.load(fp)

