# Hip_shape manipulation
This repository contains some code which allows one to manipulate the data for the Hip shape project. Full documentation can be found [here](https://adwaye.github.io/hip_shape/index.html)

## Getting started
The raw data is stored in the following folders:
TOH-Controls, TOH-DDH, TOH-FAI, UCLH-Controls, UCLH-Dysplastics in the university of bath x:drive. Each of the folders have the following structure, where only the highlighted contents are important. 

    ::
        | path
        | ├── C4
        | │   ├── Left
        | │   │   ├── C4_LPEL.mha
        | │   │   ├── C4_LPEL.vtk
        | │   │   ├── C4_LPEL.stl
        | │   └── Right
        | │   │   ├── C4_RPEL.mha
        | │   │   ├── C4_RPEL.vtk
        | │   │   ├── C4_RPEL.stl
        | │   └── landmarks.xlsx

To start the data analysis, one should clone this repository and run the following command which creates the folder <tt>data/Segmentation_and_landmarks_raw</tt>.

    foo@bar$ mkdir data/Segmentation_and_landmarks_raw


The landmarks for the socket opening and the APP is foud in the <tt>.xlsx</tt> file. To extract these landmarks and store them along with the surfaces in the related <tt>.stl</tt> files, one can run the following
    
    foo@bar$ python3 data_cleaning.py


This creates pickle files for each study under <tt>data/Segmentation_and_landmarks_processed</tt> which have the following structure:
 1. **out_dict['surface']**: is a dictionary that saves the Pelvis surfaces and has keys 'RPel' and 'LPel' for the right pelvis and left pelvis respectively. out_dict['surface']['RPel']  and out_dict['surface']['RPel'] are a dictionaries with keys.
      * 'points' : np.array shape [N1,3] of vertices.
      * 'faces'  : np.array shape [N2,3] of simplices where vertices[faces[:,i]] would be the coordinates of the i-th vertex in each triangle.
      * 'mesh_loc': the location of the raw mesh .stl file
 2. **out_dict['landmarks']**: is a dictionary containing the ladnmarks placed on the hips with keys
        * 'RASIS' : np.array shape (,3) location of the right anterior illiac spine on the triangular mesh definining the anterior pelvic plane
        * 'LASIS' : np.array shape (,3) location of the left anterior illiac spine on the triangular mesh definining the anterior pelvic plane
        * 'RTUB'  : np.array shape (,3) location of the right acetabulum on the triangular mesh definining the anterior pelvic plane
        * 'LTUB'  : np.array shape (,3) location of the left acetabulum on the triangular mesh definining the anterior
        * 'Right Ant Lat' : np.array shape (n1,3) right anterior socket crest
        * 'Right Post Lat': np.array shape (n2,3) right posterior socket crest
        * 'Left Ant Lat'  : np.array shape (n3,3) left anterior socket crest
        * 'Left Post Lat' : np.array shape (n4,3) left posterior socket crest

## Some helpful classes
1. **data_utils.HipShape**: allows the user to load the pickle files and extract the data by calling class attributes
2. **data_utils.MeshLibDecimator**: Together with the previous class, the user can reduce the size of the meshes. 
   
    
    from data_utils import *
    path1 = 'data/Segmentation_and_landmarks_processed/c4.p'
    path2 = 'data/Segmentation_and_landmarks_processed/c4.p'
    decimator = MeshLibDecimator()
    hip1 = HipData(path1,decimator)
    hip2 = HipData(path2,decimator)
    #aligning the hips by the righ pelvis:
    hip1,hip2 = ralign_2_hips(hip1,hip2,by='RPel')
    #reduce number of vertices and save the result
    hip1.decimate(max_num_faces=1000,save_path = 'data/Segmentation_and_landmarks_processed_10')


## Visualisation
One can visualise the data by running the following:
    foo@bar$ python3 visualiser_mayavi.py

There are also classes in visualiser.py that would allow one to view the data when running loops on them, which might be useful when training shape models with pytorch. The backend uses mayavi, for which a generator is required to animate the data. 



    
      