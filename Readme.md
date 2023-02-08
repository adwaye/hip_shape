# File manipulation for hip shape project

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


The landmarks for the socket opening and the APP is foud in the <tt>.xlsx</tt> file. To extract these landmarks and store them along with the surfaces in the related <tt>.stl</tt> files, one can run the  


 #. **out_dict['surface']**: is a dictionary that saves the Pelvis surfaces and has keys 'RPel' and 'LPel' for the right pelvis and left pelvis respectively. out_dict['surface']['RPel']  and out_dict['surface']['RPel'] are a dictionaries with keys.
      * 'points' : np.array shape [N1,3] of vertices.
      * 'faces'  : np.array shape [N2,3] of simplices where vertices[faces[:,i]] would be the coordinates of the i-th vertex in each triangle.
      * 'mesh_loc': the location of the raw mesh .stl file
 #. **out_dict['landmarks']**: is a dictionary containing the ladnmarks placed on the hips with keys
        * 'RASIS' : np.array shape (,3) location of the right anterior illiac spine on the triangular mesh definining the anterior pelvic plane
        * 'LASIS' : np.array shape (,3) location of the left anterior illiac spine on the triangular mesh definining the anterior pelvic plane
        * 'RTUB'  : np.array shape (,3) location of the right acetabulum on the triangular mesh definining the anterior pelvic plane
        * 'LTUB'  : np.array shape (,3) location of the left acetabulum on the triangular mesh definining the anterior
        * 'Right Ant Lat' : np.array shape (n1,3) right anterior socket crest
        * 'Right Post Lat': np.array shape (n2,3) right posterior socket crest
        * 'Left Ant Lat'  : np.array shape (n3,3) left anterior socket crest
        * 'Left Post Lat' : np.array shape (n4,3) left posterior socket crest