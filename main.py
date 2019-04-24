from sklearn import model_selection
#from tensorflow.python.estimator import keras

from mlltranspiler import MLL


# import warnings
# warnings.filterwarnings("ignore")



skmodel1 = """rf_clf  : @RandomForestClassifier with n_estimators = 10 criterion = 'entropy' 
knn_clf : @KNeighborsClassifier 2
svc_clf : @SVC with C=10000.0
rg_clf  : @RidgeClassifier 0.1
dt_clf  : @DecisionTreeClassifier with criterion='gini'
lr      : @LogisticRegression
sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr
"""

skmodel2 = """rf_clf  : @RandomForestClassifier with criterion = 'entropy' 10
knn_clf : @KNeighborsClassifier 2
svc_clf : @SVC with C=10000.0
rg_clf  : @RidgeClassifier 0.1
dt_clf  : @DecisionTreeClassifier with criterion='gini'
lr      : @LogisticRegression
sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr
"""

skmodel3 = """rf_clf  : @RandomForestClassifier 10 with criterion = 'entropy'
knn_clf : @KNeighborsClassifier 2
svc_clf : @SVC with C=10000.0
rg_clf  : @RidgeClassifier 0.1
dt_clf  : @DecisionTreeClassifier with criterion='gini'
lr      : @LogisticRegression
sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr
"""

net1 = """ 
params_conv2d := (3, 3) with padding='same'

model : seq 
+ Lambda augment_2d with input_shape=x_train.shape[1:] arguments={'rotation': 8.0, 'horizontal_flip': True} 
+ conv2d 32 params_conv2d 
+ relu
+ Conv2D 32 (3, 3) 
+ relu 
+ MaxPooling2D with pool_size=(2, 2) 
+ Dropout 0.25
+ conv2d 64 params_conv2d 
+ relu 
+ Conv2D 64 (3, 3) 
+ relu 
+ MaxPooling2D with pool_size=(2, 2) + Dropout 0.25 + flatten + dense 512 + relu
+ dropout 0.5 + dense @num_classes + softmax
"""

net2 = """ 
params_conv2d := (3, 3) with padding='same'
conv2d := Conv2D
model : conv2d 3.0 params_conv2d
"""

net3 = """ 
params_conv2d := (3, 3) with padding='same'
"""

mix = """
params_conv2d := (3, 3) with padding='same'
conv2d := Conv2D
seq := Sequential

rf_clf  : @RandomForestClassifier 10 with criterion = 'entropy'
knn_clf : @KNeighborsClassifier 2
svc_clf : @SVC with C=10000.0
rg_clf  : @RidgeClassifier 0.1
dt_clf  : @DecisionTreeClassifier with criterion='gini'
lr      : @LogisticRegression
sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

model : seq 
+ Lambda augment_2d with input_shape=x_train.shape[1:] arguments={'rotation': 8.0, 'horizontal_flip': True} 
+ conv2d 32 params_conv2d 
+ relu
+ Conv2D 32 (3, 3) 
+ relu 
+ MaxPooling2D with pool_size=(2, 2) 
+ Dropout 0.25
+ conv2d 64 params_conv2d 
+ relu 
+ Conv2D 64 (3, 3) 
+ relu 
+ MaxPooling2D with pool_size=(2, 2) + Dropout 0.25 + flatten + dense 512 + relu
+ dropout 0.5 + dense @num_classes + softmax

"""

mix_no_augment = """
params_conv2d := (3, 3) with padding='same'
conv2d := Conv2D
seq := Sequential
re := Activation 'relu'
drop := Dropout
dense := Dense
flatten := Flatten
soft := Activation 'softmax'


rf_clf  : @RandomForestClassifier 10 with criterion = 'entropy'
knn_clf : @KNeighborsClassifier 2
svc_clf : @SVC with C=10000.0
rg_clf  : @RidgeClassifier 0.1
dt_clf  : @DecisionTreeClassifier with criterion='gini'
lr      : @LogisticRegression
sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

net : seq 
+ conv2d 32 params_conv2d 
+ re
+ Conv2D 32 (3, 3) 
+ re 
+ MaxPooling2D with pool_size=(2, 2) 
+ Dropout 0.25
+ conv2d 64 params_conv2d 
+ re 
+ Conv2D 64 (3, 3) 
+ re 
+ MaxPooling2D with pool_size=(2, 2) + Dropout 0.25 + flatten + dense 512 + re
+ drop 0.5 + dense 512 + soft

"""



# simple_net = """
# params_conv2d := (3, 3) with padding='same'
# conv2d := Conv2D
# seq := Sequential
# relu := Activation 'relu'
# drop := Dropout
# dense := Dense
# flatten := Flatten
# soft := Activation 'softmax'
# ANN := seq
#
# padding have 'same' or 'valid'
#
# criterion have 'gini' or 'entropy'
#
# classifier rf_clf  : @RandomForestClassifier 10 entropy
# knn_clf : @KNeighborsClassifier 2
# svc_clf : @SVC with C=10000.0
# rg_clf  : @RidgeClassifier 0.1
# dt_clf  : @DecisionTreeClassifier gini
# lr      : @LogisticRegression
# sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr
#
# net : ANN
# + Conv2D 32 (3, 3) with input_shape=(100, 100, 3)
# + relu
# + flatten
# + Dense 256
# + relu
# + Dropout 0.5
# + Dense 10 with activation='softmax'
# """

# skmodel4 = """
#
# criterion have 'gini' or 'entropy'
#
# rf_clf  : @RandomForestClassifier 10 entropy
# knn_clf : @KNeighborsClassifier 2
# svc_clf : @SVC with C=10000.0
# rg_clf  : @RidgeClassifier 0.1
# dt_clf  : @DecisionTreeClassifier gini
# lr      : @LogisticRegression
# classifier sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr
#
# """



import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def get_data():
    iris_dataset = pd.read_csv("iris.csv")
    train, test = iris_dataset.iloc[:, 0:4], iris_dataset.iloc[:, 4]

    encoder_object = LabelEncoder()
    test = encoder_object.fit_transform(test)

    return train, test

def test_sk_1():
    print("----------------------TEST_SK_1----------------------------")
    sclf = MLL(skmodel4).last_model()

    # train, test = get_data()
    #
    # sclf.fit(train, test)
    #
    # scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
    # print(scores.mean(), scores.std())

#deve essere tradotta
another_net="""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
"""


#concat invece che essere una parola del linguaggio potrebbe essere una funzione python interna che andiamo a chiamare

inception = """
conv2d := Conv2D
seq := Sequential
re := Activation 'relu'
drop := Dropout
dense := Dense
flatten := Flatten
soft := Activation 'softmax'

c2d32 := Conv2D 32 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
c2d48 := Conv2D 48 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
c2d64 := Conv2D 64 3 3 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
c2d96 := Conv2D 96 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
c2d192 := Conv2D 192 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re


#layer di input, riceve in input x
stem : x + c2d32 + c2d32 + c2d64

biforcazione1 : stem + m2d
biforcazione2 : stem + c2d96

stem : concat biforcazione1 biforcazione2

biforcazione1 : stem + c2d64 + c2d96
biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

stem : concat biforcazione1 biforcazione2

biforcazione1 : stem + c2d192
biforcazione2 : stem + m2d

stem : concat biforcazione1 biforcazione2

stem : stem + re

#layer A, riceve in input x

biforcazione1 : x + c2d32
biforcazione2 : x + c2d32 + c2d32
biforcazione3 : x + c2d32 + c2d48 + c2d64

A : concat biforcazione1 biforcazione2 biforcazione3

A : A + c2d384
A : concat A x
A : A + re

#layer redA, riceve in input x

m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

biforcazione1 : x + m2d
biforcazione2 : x + c2d384
biforcazione3 : x + c2d256 + c2d256 + c2d28422

redA : concat biforcazione1 biforcazione2 biforcazione3 

#layer B, riceve in input x

#da finire

"""

# inception_uncomm = """
# conv2d := Conv2D
# seq := Sequential
# re := Activation 'relu'
# drop := Dropout
# dense := Dense
# flatten := Flatten
# soft := Activation 'softmax'
#
# c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# c2d48 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
# m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
# c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# c2d192 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# c2d384 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
#
# stem : seq + InputLayer (100, 100, 3) + c2d32 + c2d32 + c2d64
#
# biforcazione1 : stem + m2d
# biforcazione2 : stem + c2d96
#
# stem : Concatenate biforcazione1 biforcazione2
#
# biforcazione1 : stem + c2d64 + c2d96
# biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96
#
# stem : concatenate biforcazione1 biforcazione2
#
# biforcazione1 : stem + c2d192
# biforcazione2 : stem + m2d
#
# stem : concatenate biforcazione1 biforcazione2
#
# stem : stem + re
#
# biforcazione1 : stem + c2d32
# biforcazione2 : stem + c2d32 + c2d32
# biforcazione3 : stem + c2d32 + c2d48 + c2d64
#
# A : concatenate biforcazione1 biforcazione2 biforcazione3
#
# A : A + c2d384
# A : concatenate A stem
# A : A + re
#
# m2d := MaxPooling2D 3, 3 with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
# c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
# c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re
#
# biforcazione1 : A + m2d
# biforcazione2 : A + c2d384
# biforcazione3 : A + c2d256 + c2d256 + c2d3822
#
# redA : concatenate biforcazione1 biforcazione2 biforcazione3
#
# """

# inception_uncomm_simpler = """
# seq := Sequential
# re := Activation 'relu'
#
# c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
#
# stem : seq
#         + c2d32
#         + c2d32
# """

#
# inception_layer = """
# c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
# m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
# re := Activation 'relu'
#
# inception_stem : c2d32 | c2d32 + c2d32 | c2d32 + c2d32 + c2d32 | m2d
# """



def test_inception_1():
    print("----------------------TEST_NET_1----------------------------")
    #mll = MLL(inception_layer)


    
def test_net_1():
    print("----------------------TEST_NET_1----------------------------")
    # print("-------------mlltranspiler result--------------")
    #mll = MLL(simple_net)
    # print("----------------main result--------------------")
    #net = mll.last_model()
    # #net2 = mll.models
    # print(net)
    # print(type(net))
    # print("----------------models result------------------")
    # print(mll.models)
    # print("----------------macros result------------------")
    # print(mll.macros)
    # print("----------------string result------------------")
    # print(mll.string[:100]+"........")
    # print("-------------------fitting---------------------")


    # import numpy as np
    # import keras
    # from keras.models import Sequential
    # from keras.layers import Dense, Dropout, Flatten
    # from keras.layers import Conv2D, MaxPooling2D
    # from keras.optimizers import SGD
    #
    # # Generate dummy data
    # x_train = np.random.random((100, 100, 100, 3))
    # y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    # x_test = np.random.random((20, 100, 100, 3))
    # y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    #
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # net.compile(loss='categorical_crossentropy', optimizer=sgd)
    #
    # net.fit(x_train, y_train, batch_size=32, epochs=10)
    # score = net.evaluate(x_test, y_test, batch_size=32)
    #
    # print(net.summary())


def test_inception():
    mll = MLL(inception_uncomm)
    # net = mll.last_model()
    #
    # print(type(net))
    #
    # import numpy as np
    # import keras
    # from keras.optimizers import SGD
    #
    # # Generate dummy data
    # x_train = np.random.random((100, 100, 100, 3))
    # y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    # x_test = np.random.random((20, 100, 100, 3))
    # y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    #
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # net.compile(loss='categorical_crossentropy', optimizer=sgd)
    #
    # net.fit(x_train, y_train, batch_size=32, epochs=10)
    # score = net.evaluate(x_test, y_test, batch_size=32)
    #
    # print(net.summary())


def main():

    print("------------------------------------------------")
    test_sk_1()
    print("------------------------------------------------")
    test_inception()
    print("------------------------------------------------")
    test_inception_1()


if __name__ == '__main__':
    main()