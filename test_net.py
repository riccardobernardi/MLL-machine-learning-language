# from keras import layers
#
# # print(list(layers.__dict__))
#
# arr=[]
# keras_layers = set()
#
# for k in layers.__dict__.keys():
#     if "__" not in k and k[0].islower() and k != "K":
#         arr+=[k]
#
# for i in arr:
#     print(i)

models={}

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
from keras.layers import Concatenate
from keras.layers import concatenate
from keras.layers import InputLayer

#macro: Conv2D
#macro: Sequential
#macro: Activation 'relu'
#macro: Dropout
#macro: Dense
#macro: Flatten
#macro: Activation 'softmax'
#macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
#macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
models['stem']=Sequential()
models['stem'].add(InputLayer((100,100,3)))
models['stem'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['stem'].add(Activation('relu'))
models['stem'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['stem'].add(Activation('relu'))
models['stem'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['stem'].add(Activation('relu'))
models['biforcazione1']=models['stem']
models['biforcazione1'].add(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))
models['biforcazione2']=models['stem']
models['biforcazione2'].add(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['stem']=concatenate([models['biforcazione1'],models['biforcazione2']],axis=-1)
models['biforcazione1']=models['stem']
models['biforcazione1'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione1'].add(Activation('relu'))
models['biforcazione1'].add(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione1'].add(Activation('relu'))
models['biforcazione2']=models['stem']
models['biforcazione2'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione2'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione2'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione2'].add(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['stem']=Concatenate([models['biforcazione1'],models['biforcazione2']])
models['biforcazione1']=models['stem']
models['biforcazione1'].add(Conv2D(192,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione1'].add(Activation('relu'))
models['biforcazione2']=models['stem']
models['biforcazione2'].add(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))
models['stem']=concatenate([models['biforcazione1'],models['biforcazione2']])
models['stem']=models['stem']
models['stem'].add(Activation('relu'))
models['biforcazione1']=models['stem']
models['biforcazione1'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione1'].add(Activation('relu'))
models['biforcazione2']=models['stem']
models['biforcazione2'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione2'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione3']=models['stem']
models['biforcazione3'].add(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['biforcazione3'].add(Conv2D(48,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['biforcazione3'].add(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['A']=concatenate([models['biforcazione1'],models['biforcazione2'],models['biforcazione3']])
models['A']=models['A']
models['A'].add(Conv2D(384,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['A'].add(Activation('relu'))
models['A']=concatenate([models['A'],models['stem']])
models['A']=models['A']
models['A'].add(Activation('relu'))
#macro: MaxPooling2D 3, 3 with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
#macro: Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#macro: Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
#macro: Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re
models['biforcazione1']=models['A']
models['biforcazione1'].add(MaxPooling2D(3,3,strides=(2,2),border_mode='valid',dim_ordering='tf'))
models['biforcazione2']=models['A']
models['biforcazione2'].add(Conv2D(384,3,3,subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))
models['biforcazione2'].add(Activation('relu'))
models['biforcazione3']=models['A']
models['biforcazione3'].add(Conv2D(384,1,1,subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['biforcazione3'].add(Conv2D(384,1,1,subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['biforcazione3'].add(Conv2D(384,1,1,subsample=(2,2),init='he_normal',border_mode='same',dim_ordering='tf'))
models['biforcazione3'].add(Activation('relu'))
models['redA']=concatenate([models['biforcazione1'],models['biforcazione2'],models['biforcazione3']])
