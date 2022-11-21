from threading import Condition,Thread
import time
import random
import os
from visualiser import HipDataSource, MayaviObserver
items=[]

hip_data_source = HipDataSource(pickle_path='/home/adwaye/PycharmProjects/hip_shape/data'
                                            '/Segmentation_and_landmarks_downsample_10/TOH - Controls/C4.p',
                                decimator=None)
pickle_loc = '/home/adwaye/PycharmProjects/hip_shape/data/Segmentation_and_landmarks_downsample_10/TOH - Controls/'
files = [os.path.join(pickle_loc,f) for f in os.listdir(pickle_loc)]

# for k,pickle_path in enumerate(files):
#     time.sleep(5)
#     if k < 10:
#         print(pickle_path)
#         hip_data_source.set_data(pickle_path=pickle_path)
#         hip_data_source.notifyObservers()
#
#     else:
#         break
def produce(c):

    for k,pickle_path in enumerate(files):
        c.acquire() #Step 1.1
        item=hip_data_source.set_data(pickle_path) #Step 1.2
        print("Producer Producing Item:", hip_data_source.pickle_path)
        #items.append(item) #Step 1.3
        print("Producer giving Notification")
        c.notify() #Step 1.4
        c.release() #Step 1.5
        time.sleep(5)
def consume(c):
    observer = MayaviObserver()
    while True:
        c.acquire() #Step 2.1
        print("Consumer waiting for update")
        c.wait() #Step 2.2
        print("Consumer consumed the item", hip_data_source.pickle_path) #Step 2.3
        observer.set_data(hip_data_source.get_data())
        #observer.display_data()
        c.release() #Step 2.4
        time.sleep(5)



c=Condition()
t1=Thread(target=consume, args=(c,))
t2=Thread(target=produce, args=(c,))
t1.start()
t2.start()