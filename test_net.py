models = {}

######################################################################
#                            STARTING MLL                            #
######################################################################
# 1. model = MLL('PROGRAM.MLL')                                      #
# 2. model.start() -> to pass from MLL to python                     #
# 3. model.get_string() -> to get python code of your program        #
# 4. model.execute() -> to run python code of your program           #
# 5. clf = model.last_model() -> to get last model of your program   #
# 6. MLL() -> to get this window                                     #
#                                                                    #
#                                    student: Bernardi Riccardo      #
#                                    supervisor: Lucchese Claudio    #
#                                    co-supervisor: Spanò Alvise     #
######################################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.model_selection import *
from mlxtend.classifier import StackingClassifier
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import InputLayer
from keras.layers import AveragePooling2D

#macro: Conv2D
#macro: Sequential
#macro: Activation relu
#macro: Dropout
#macro: Dense
#macro: Flatten
#macro: Activation 'softmax'
#macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
#macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
#macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
#macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
# Input layer
models['x']=Input(shape=(32,32,3))
# Layer stem di entrata dell input

def stem1(x):
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(a)
	a=(Activation('relu'))(a)
	a=(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(a)
	a=(Activation('relu'))(a)
	return a

models['x']=stem1(models['x'])

def stem2(x):
	a=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(x)
	b=(Activation('relu'))(b)
	x=concatenate([b,a])
	return b

models['x']=stem2(models['x'])

def stem3(x):
	a=(Conv2D(64,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	a=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(a)
	a=(Activation('relu'))(a)
	b=(Conv2D(64,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	b=(Activation('relu'))(b)
	b=(Conv2D(64,(7,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(b)
	b=(Activation('relu'))(b)
	b=(Conv2D(64,(1,7),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(b)
	b=(Activation('relu'))(b)
	b=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(b)
	b=(Activation('relu'))(b)
	x=concatenate([b,a])
	return b

models['x']=stem3(models['x'])

def stem4(x):
	a=(Conv2D(192,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	b=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(x)
	x=concatenate([b,a])
	return b

models['x']=stem4(models['x'])

def stem5(x):
	a=(Activation('relu'))(x)
	return a

models['x']=stem5(models['x'])
# layer A

def incA1(x):
	a=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	b=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	b=(Activation('relu'))(b)
	b=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(b)
	b=(Activation('relu'))(b)
	c=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	c=(Activation('relu'))(c)
	c=(Conv2D(48,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(c)
	c=(Activation('relu'))(c)
	c=(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(c)
	c=(Activation('relu'))(c)
	x=concatenate([c,b,a])
	return c

models['x']=incA1(models['x'])
models['shortcut']=models['x']()

def incA2(x):
	a=(Conv2D(384,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	b=(shortcut())(x)
	x=concatenate([b,a])
	return b

models['x']=incA2(models['x'])

def incA3(x):
	a=(Activation('relu'))(x)
	return a

models['x']=incA3(models['x'])
# nn funziona dobbiamo poter fare dag all interno di altri dag
# la merge sum non e permessa
# i border sono messi sbagliati nelle macro