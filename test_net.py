models = {}

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
from keras.models import Input
from keras.backend import shape
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge

def assign (x):
    return x

#macro: Conv2D
 #macro: Sequential
#macro: Activation relu
#macro: Dropout
#macro: Dense
#macro: Flatten
#macro: Activation 'softmax'
#macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
#macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
#macro: MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
#macro: Conv2D 256 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 256 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
#macro: Conv2D 384 (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
#macro: MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
#  Input layer
models['x']=Input(shape=(32,32,3))
#  Layer stem di entrata dell input

def stem1(x):
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(a)
	a=(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(a)
	return a

models['x']=stem1(models['x'])

def stem2(x):
	a=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	b=merge([a,b],'concat')
	c=(Conv2D(64,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	c=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(c)
	d=(Conv2D(64,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	d=(Conv2D(64,(7,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(d)
	d=(Conv2D(64,(1,7),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(d)
	d=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(d)
	d=merge([c,d],'concat')
	e=(Conv2D(192,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(d)
	f=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(d)
	f=merge([e,f],'concat')
	return f

models['x']=stem2(models['x'])

def stem5(x):
	a=(Activation('relu'))(x)
	return a

models['x']=stem5(models['x'])
#  layer A
models['shortcut']=assign(models['x'])

def incA1(x):
	a=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	b=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(b)
	c=(Conv2D(32,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(48,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=merge([a,b,c],'concat')
	d=(Conv2D(384,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	e=(assign(models['shortcut']))
	e=merge([d,e],'concat')
	return e

# l ultima concat qui sopra sarebbe una sum
# bisogna definire sum
models['x']=incA1(models['x'])
#  la parte del concat o sum non e presente nei precedenti tests
#  dovremmo fare una versione di questo test piu corto

def incA1_end(x):
	a=(Activation('relu'))(x)
	return a

models['x']=incA1_end(models['x'])
#  nn funziona dobbiamo poter fare dag all interno di altri dag
#  la merge sum non e permessa

def incA1_red(x):
	a=(MaxPooling2D((3,3),strides=(2,2),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(384,(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(256,(1,1),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(x)
	c=(Conv2D(256,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf',activation='relu'))(c)
	c=(Conv2D(384,(3,3),subsample=(2,2),init='he_normal',border_mode='valid',dim_ordering='tf',activation='relu'))(c)
	c=merge([a,b,c],'concat')
	return c

models['x']=incA1_red()
