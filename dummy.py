
import random,logging
import concurrent.futures
import threading,time
from visualiser import HipDataSource
import os
import queue
import jax.numpy as jnp

from mayavi import mlab
# from mayavi.api import Engine
# from tvtk.api import tvtk
import numpy as np
SENTINEL = object()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt





def init_vis():
    # Creates a new Engine, starts it and creates a new scene
    # engine = Engine()
    # engine.start()
    # engine.new_scene()

    # Initialise Plot

    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C8.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    vertices = points - np.mean(points,axis=0,keepdims=True)



    # vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,2,1]])
    # T = np.array([[0,1,2],[1,2,3]])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    tri_plot= ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces,edgecolor=[[0,0,0]],
                             linewidth=1.0,
                    alpha=0.0,
                    shade=False)

    plt.show()


    return tri_plot
#@mlab.animate(delay=500,ui=False)



def jax_func(data):
    noise = jnp.array(np.random.randn(*data.shape)*10)
    data = jnp.add(data,noise)
    return data

def producer_old(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        pipeline.set_message(message, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")


def producer(pipeline):
    """Pretend we're getting a message from the network."""
    hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                                '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                    decimator=None)
    points,faces = hip_data_source.get_data()
    for index in range(0):
        points = jax_func(points)
        message = random.randint(1, 101)
        logging.info(f'Producer got message: {points}')
        pipeline.set_message(points, "Producer")

    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")

def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    tri_plot = init_vis()
    mlab.show()
    message = 0
    while message is not SENTINEL:
        points = pipeline.get_message("Consumer")
        if message is not SENTINEL:
            logging.info("Consumer storing message: %s", points)
            tri_plot.set_verts(points)




class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self, name):
        self.consumer_lock.acquire()
        message = self.message
        self.producer_lock.release()
        return message

    def set_message(self, message, name):
        self.producer_lock.acquire()
        self.message = message
        self.consumer_lock.release()







if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.DEBUG)

    pipeline = Pipeline()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)

    #tri_plot = init_vis()