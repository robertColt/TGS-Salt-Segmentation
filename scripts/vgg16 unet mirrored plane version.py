#!/usr/bin/env python
# coding: utf-8

# In[10]:


#imports necessary 

import numpy as np     # linear algebra library
import pandas as pd    
import tensorflow as tf

from random import randint

import matplotlib.pyplot as plt    # data visualization
plt.style.use('seaborn-white')
import seaborn as sns              # data visualization
sns.set_style("white")

from sklearn.model_selection import train_test_split        # function used for splitting the data 
                                                            # in training and validation, with the
                                                            # same distribution (based on salt coverage)

from skimage.transform import resize         

# keras specifics

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Add, Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras import backend as K
# import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras import optimizers

from tqdm import tqdm_notebook
from tensorflow.compat.v1.metrics import mean_iou
from keras.layers.core import Lambda


# In[20]:


location = "C://Users//oana.sabadas//PycharmProjects//TGS//tgs-salt-identification-challenge//Numpy//splitted//128"   
#location of the files to be read


# In[21]:


y_train = np.load("{}//y_train.npy".format(location),allow_pickle=True)
x_train = np.load("{}//x_train.npy".format(location),allow_pickle=True)


y_valid = np.load("{}//y_valid.npy".format(location),allow_pickle=True)
x_valid = np.load("{}//x_valid.npy".format(location),allow_pickle=True)


ids_train = np.load("{}//ids_train.npy".format(location),allow_pickle=True)
ids_valid = np.load("{}//ids_valid.npy".format(location),allow_pickle=True)

            
cov_train = np.load("{}//cov_train.npy".format(location),allow_pickle=True)
cov_valid = np.load("{}//cov_test.npy".format(location),allow_pickle=True)


depth_train = np.load("{}//depth_train.npy".format(location),allow_pickle=True)
depth_valid = np.load("{}//depth_test.npy".format(location),allow_pickle=True)


# In[22]:


x_train.shape     # check te shape to see if everything is alright


# In[23]:


def iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# In[24]:


def BatchActivation(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# In[25]:


# Build U-Net model
inputs = Input((128, 128, 1))
s = Lambda(lambda x: x / 255) (inputs)

# block 1 of convolution ---- 2 conv layers followed by max pooling
"""
Conv2D(number_of_filters,(filtersize,filter_size))

"""

c0 = Conv2D(4, (3, 3), padding='same') (s)
c0 = BatchActivation(c0)
c0 = Conv2D(4, (3, 3), padding='same') (c0)
c0 = BatchActivation(c0)
p0 = MaxPooling2D((2, 2)) (c0)

#block 2 of convolution ---- 2 conv layers followed by max pooling
c1 = Conv2D(8, (3, 3), padding='same') (p0)
c1 = BatchActivation(c1)
c1 = Conv2D(8, (3, 3), padding='same') (c1)
c1 = BatchActivation(c1)
p1 = MaxPooling2D((2, 2)) (c1)

#block 3 of convolution   ---- 3conv layers, max pooling
c2 = Conv2D(16, (3, 3), padding='same') (p1)
c2 = BatchActivation(c2)
c2 = Conv2D(16, (3, 3), padding='same') (c2)
c2 = BatchActivation(c2)
c2 = Conv2D(16, (3, 3), padding='same') (c2)
c2 = BatchActivation(c2)
p2 = MaxPooling2D((2, 2)) (c2)


# bock 4 of convolution ----- 3conv layers followed by max pooling
c3 = Conv2D(32, (3, 3), padding='same') (p2)
c3 = BatchActivation(c3)
c3 = Conv2D(32, (3, 3), padding='same') (c3)
c3 = BatchActivation(c3)
c3 = Conv2D(32, (3, 3), padding='same') (c3)
c3 = BatchActivation(c3)
p3 = MaxPooling2D((2, 2)) (c3)



#block 5 of convolution  --- 2 conv layers max pooling
c4 = Conv2D(64, (3, 3), padding='same') (p3)
c4 = BatchActivation(c4)
c4 = Conv2D(64, (3, 3), padding='same') (c4)
c4 = BatchActivation(c4)
c4 = Conv2D(64, (3, 3), padding='same') (c4)
c4 = BatchActivation(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


# middle layer    --- learns 128 filters
c5 = Conv2D(128, (3, 3), padding='same') (p4)
c5 = BatchActivation(c5)
c5 = Conv2D(128, (3, 3), padding='same') (c5)
c5 = BatchActivation(c5)


# first layer of decoder ( concatenates with 5th layer of vgg convolution)
u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), padding='same') (u6)
c6 = BatchActivation(c6)
c6 = Conv2D(64, (3, 3), padding='same') (c6)
c6 = BatchActivation(c6)
c6 = Conv2D(64, (3, 3), padding='same') (c6)
c6 = BatchActivation(c6)



# second layer of decoder ( concatenates with 4th layer of vgg convolution)
u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), padding='same') (u7)
c7 = BatchActivation(c7)
c7 = Conv2D(32, (3, 3), padding='same') (c7)
c7 = BatchActivation(c7)
c7 = Conv2D(32, (3, 3), padding='same') (c7)
c7 = BatchActivation(c7)




# third layer of decoder ( concatenates with 3rd layer of vgg convolution)
u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), padding='same') (u8)
c8 = BatchActivation(c8)
c8 = Conv2D(16, (3, 3), padding='same') (c8)
c8 = BatchActivation(c8)
c8 = Conv2D(16, (3, 3), padding='same') (c8)
c8 = BatchActivation(c8)



# forth layer of decoder ( concatenates with 2nd layer of vgg convolution)
u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), padding='same') (u9)
c9 = BatchActivation(c9)
c9 = Conv2D(8, (3, 3), padding='same') (c9)
c9 = BatchActivation(c9)



# first layer of decoder (concatenates with input layer from vgg convolution)
u10 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c0], axis=3)
c10 = Conv2D(4, (3, 3), padding='same') (u10)
c10 = BatchActivation(c10)
c10 = Conv2D(4, (3, 3), padding='same') (c10)
c10 = BatchActivation(c10)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c10)


# In[26]:


model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[iou])
model.summary()


# In[27]:


model.save("models/unet_vgg16_128_adam_binary-crossentropy_iou-metric.h5")


# In[28]:


num_epochs = 100
to_be_saved = "models/model_128.h5"                  # <---- change here the name of the file to be saved

early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(to_be_saved,monitor='iou', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='iou', factor=0.1, patience=5, min_lr=0.0001, verbose=1)

batch_size = 32


# In[ ]:


history = model.fit(x_train,
                     y_train,
                     validation_data=[x_valid,
                                      y_valid],
                     epochs=num_epochs,
                     batch_size=batch_size,
                     callbacks=[early_stopping, 
                               model_checkpoint,
                               reduce_lr]
                    )

