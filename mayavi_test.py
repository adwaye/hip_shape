import numpy as np
from mayavi import mlab
from stl import mesh
import os
import matplotlib.pyplot as plt

def main():
    file_name = "/home/adwaye/PycharmProjects/hip_shape/data/UCLH-Controls/97940593/Right/97940593_RPEL.stl"
    my_mesh = mesh.Mesh.from_file(file_name)
    # x = my_mesh.x
    # y = my_mesh.y
    # z = my_mesh.z
    # mlab.points3d(x, y, z)
    # mlab.show()
    m_data = mlab.pipeline.open(file_name)
    mlab.pipeline.surface(m_data)
    mlab.show()


    import os
    from os.path import join

    # Enthought library imports
    from mayavi import mlab

    ### Download the bunny data, if not already on disk ############################
    if not os.path.exists('bunny.tar.gz'):
        # Download the data
        try:
            from urllib import urlopen
        except ImportError:
            from urllib.request import urlopen
        print("Downloading bunny model, Please Wait (3MB)")
        opener = urlopen(
                    'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz')
        open('bunny.tar.gz', 'wb').write(opener.read())

    # Extract the data
    import tarfile
    bunny_tar_file = tarfile.open('bunny.tar.gz')
    try:
        os.mkdir('bunny_data')
    except:
        pass
    bunny_tar_file.extractall('bunny_data')
    bunny_tar_file.close()

    # Path to the bunny ply file
    bunny_ply_file = join('bunny_data', 'bunny', 'reconstruction', 'bun_zipper.ply')

    # Render the bunny ply file
    mlab.pipeline.surface(mlab.pipeline.open(bunny_ply_file))
    mlab.show()

    import shutil
    shutil.rmtree('bunny_data')

if __name__=='__main__':
    main()