import argparse	
import os

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)

from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import videoto3d
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.models import load_model





def loaddata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):

    
    files = os.listdir(video_dir)    
    print("Inside Load Data\n")
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for filename in files:
    
        pbar.update(1)
        
        if filename == '.DS_Store':
            continue
        
        name = os.path.join(video_dir, filename)
        print("Video_dir + filename ",name)

        for v_files in os.listdir(name):

            v_file_path = os.path.join(name, v_files)
            print("v_file_path ",v_file_path)
            label = filename
            print("Label ",label)
            if label not in labellist:
                
                if len(labellist) >= nclass:
                    continue

                labellist.append(label)
            
            labels.append(label)
            
            X.append(vid3d.video3d(v_file_path, color=color, skip=skip))

    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels



img_rows, img_cols, frames = 32, 32, 10
channel = 3 

vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)

x, y = loaddata('Test', vid3d, 10,
                'Output', 3, True)
print("Inside Else")
X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))

X = X.astype('float32')


model = load_model('d_3dcnnmodel-36-0.97.hd5')
print(model.summary())



y_pred  = model.predict(X)


print(y_pred)

print(np.argmax(y_pred))
