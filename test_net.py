import keras
import sklearn
import mlxtend
from keras.backend import conv2d
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.backend import flatten
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import InputLayer
from keras.backend import sum
from keras.layers import merge
from keras.layers import merge
from keras.layers import merge

def assign (x):
    return x

models = {}

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
models['x']=InputLayer((100,100,3))

def stem(x):
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(x)
	a=(Activation('relu'))(a)
	a=(Conv2D(32,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(a)
	a=(Activation('relu'))(a)
	a=(Conv2D(64,(3,3),subsample=(1,1),init='he_normal',border_mode='same',dim_ordering='tf'))(a)
	a=(Activation('relu'))(a)
	return a

models['x']=stem(models['x'])

def stem2(x):
	a=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(x)
	b=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(x)
	b=(Activation('relu'))(b)
	right=merge([a,b],'concat')
	c=(MaxPooling2D((3,3),strides=(1,1),border_mode='valid',dim_ordering='tf'))(b)
	d=(Conv2D(96,(3,3),subsample=(1,1),init='he_normal',border_mode='valid',dim_ordering='tf'))(b)
	d=(Activation('relu'))(d)
	left=merge([c,d],'concat')
	d=merge([right,left],'sum')
	return d

models['x']=stem2(models['x'])
# le concat nested senza parametri producono le lettere prima della freccia
# l ultima concat con paramteri produce x