models = {}

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_HOME'] = '/usr/local/cuda-7.5'

# In[2]:

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.engine import Input, Model
from keras.layers import merge
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time

# In[3]:

nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 3, 1))
x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 3, 1))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# # inception-resnet-v2
#
# http://arxiv.org/pdf/1602.07261v1.pdf

# In[8]:

# we reduce # filters by factor of 8 compared to original inception-v4
nb_filters_reduction_factor = 8

img_rows, img_cols = 32, 32
img_channels = 3

inputs = Input(shape=(img_rows, img_cols, img_channels))


def fant(x):
	return x // nb_filters_reduction_factor







import keras
import sklearn
import mlxtend
from keras.backend import conv2d
from keras.layers import Conv2D
from keras.models import Sequential
from keras.backend import relu
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.backend import flatten
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.backend import sum
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge


def assign(x):
	return x


# macro: Conv2D
# macro: Sequential
# macro: Activation relu
# macro: Dropout
# macro: Dense
# macro: Flatten
# macro: Activation 'softmax'
# macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 128 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 160 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 160 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 192 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 224 (1, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 256 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 256 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 256 (3, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 288 (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 288 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 320 (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
# macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'
# macro: Conv2D 384 (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
# macro: Conv2D 1154 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'
# macro: Conv2D 2048 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'
# macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
# macro: MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
# macro: MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
# macro: MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
#  Input layer
models['x']=assign(inputs)
#  Layer stem di entrata dell input

def stem1(x):
	a=(Conv2D(fant(32),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	a=(Conv2D(fant(32),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(a)
	a=(Conv2D(fant(64),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(a)
	return a

models['x']=stem1(models['x'])

def stem2(x):
	a=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(fant(96),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	b=merge([a,b],'concat')
	c=(Conv2D(fant(64),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	c=(Conv2D(fant(96),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(c)
	d=(Conv2D(fant(64),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	d=(Conv2D(fant(64),(7,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(d)
	d=(Conv2D(fant(64),(1,7),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(d)
	d=(Conv2D(fant(96),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(d)
	d=merge([c,d],'concat')
	e=(Conv2D(fant(192),(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(d)
	f=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(d)
	f=merge([e,f],'concat')
	g=(Activation('relu'))(f)
	return g

models['x']=stem2(models['x'])
#  layer A
models['shortcut']=assign(models['x'])

def incA(x):
	a=(Conv2D(fant(32),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(32),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(32),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	c=(Conv2D(fant(32),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(fant(48),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=(Conv2D(fant(64),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=merge([a,b,c],'concat')
	d=(assign(models['shortcut']))
	e=(Conv2D(fant(384),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='linear'))(c)
	e=merge([d,e],'concat')
	f=(Activation('relu'))(e)
	return f

models['x']=incA(models['x'])

def incA_red(x):
	a=(MaxPooling2D((3,3),strides=(2,2),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(fant(384),(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(fant(256),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(fant(256),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=(Conv2D(fant(384),(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(c)
	c=merge([a,b,c],'concat')
	return c

models['x']=incA_red(models['x'])
# layer B
models['shortcut']=assign(models['x'])

def incB(x):
	a=(Conv2D(fant(192),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(128),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(160),(1,7),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	b=(Conv2D(fant(192),(7,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	b=merge([a,b],'concat')
	c=(assign(models['shortcut']))
	d=(Conv2D(fant(1154),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='linear'))(b)
	d=merge([c,d],'sum')
	e=(Activation('relu'))(d)
	return e

models['x']=incB(models['x'])

def incB_red(x):
	a=(MaxPooling2D((3,3),strides=(2,2),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(fant(256),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(288),(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(b)
	c=(Conv2D(fant(256),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(fant(288),(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(c)
	d=(Conv2D(fant(256),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	d=(Conv2D(fant(288),(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(d)
	d=(Conv2D(fant(320),(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(d)
	d=merge([a,b,c,d],'concat')
	return d

models['x']=incB_red(models['x'])
models['shortcut']=assign(models['x'])

def incC(x):
	a=(Conv2D(fant(192),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(192),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(fant(224),(1,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	b=(Conv2D(fant(256),(3,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	b=merge([a,b],'concat')
	c=(assign(models['shortcut']))
	d=(Conv2D(fant(2048),(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='linear'))(b)
	d=merge([c,d],'sum')
	e=(Activation('relu'))(d)
	return e

models['x']=incC(models['x'])


x= models['x']















x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

# In[10]:

model.summary()

# In[11]:

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

# In[12]:

batch_size = 128
nb_epoch = 10
data_augmentation = True

# Model saving callback
# checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

if not data_augmentation:
	print('Not using data augmentation.')
	history = model.fit(x_train, y_train,
						batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
						validation_data=(x_test, y_test), shuffle=True,
						callbacks=[])
else:
	print('Using real-time data augmentation.')

	# realtime data augmentation
	datagen_train = ImageDataGenerator(
		featurewise_center=False,
		samplewise_center=False,
		featurewise_std_normalization=False,
		samplewise_std_normalization=False,
		zca_whitening=False,
		rotation_range=0,
		width_shift_range=0.125,
		height_shift_range=0.125,
		horizontal_flip=True,
		vertical_flip=False)
	datagen_train.fit(x_train)

	# fit the model on the batches generated by datagen.flow()
	history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
								  samples_per_epoch=x_train.shape[0],
								  nb_epoch=nb_epoch, verbose=1,
								  validation_data=(x_test, y_test),
								  callbacks=[])





