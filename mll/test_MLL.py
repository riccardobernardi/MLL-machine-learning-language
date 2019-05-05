from unittest import TestCase

from lark import Tree
from sklearn import model_selection

from mll.mlltranspiler import MLL

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import warnings

from mll.utils import list_types, get_keras_layers

warnings.filterwarnings("ignore")

import numpy as np
import keras
from keras.optimizers import SGD


def get_data():
    iris_dataset = pd.read_csv("iris.csv")
    train, test = iris_dataset.iloc[:, 0:4], iris_dataset.iloc[:, 4]

    encoder_object = LabelEncoder()
    test = encoder_object.fit_transform(test)

    return train, test


class TestMLL(TestCase):

    def test_new_only_stem(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211 := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411 := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417 := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471 := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311 := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611 := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411 := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input
        
        x :
            | c2d3233 + c2d3233 + c2d6433

        stem :
            | m2d3311
            | c2d9633
            | concat
            | c2d6411 + c2d9633
            | c2d6411 + c2d6471 + c2d6417 + c2d9633
            | concat
            | c2d19233
            | m2d3311
            | concat
            | relu

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_only_stem(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211 := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411 := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417 := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471 := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311 := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611 := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411 := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        stem1 :
            | c2d3233 + c2d3233 + c2d6433

        x : stem1 x

        stem2 :
            | m2d3311
            | c2d9633

        x : stem2 x

        stem3 :
            | c2d6411 + c2d9633
            | c2d6411 + c2d6471 + c2d6417 + c2d9633

        x : stem3 x

        stem4 :
            | c2d19233
            | m2d3311

        x : stem4 x

        stem5 :
            | relu

        x : stem5 x
        
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_half_complete_inception_more_complex(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        
        # Input layer
        
        x : Input with shape = (32,32,3)
        
        # Layer stem di entrata dell input
        
        stem1 :
            | c2d3233v + c2d3233v + c2d6433s
        
        x : stem1 x
        
        stem2 :
            | m2d3311v
            | c2d9633v
            
        x : stem2 x
        
        stem3 :
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            
        x : stem3 x
        
        stem4 :
            | c2d19233v
            | m2d3311v
            
        x : stem4 x
        
        stem5 : 
            | relu
            
        x : stem5 x
        
        # layer A
        
        shortcut : assign x
        
        incA1 :
            | c2d3211s
            | c2d3211s + c2d3233s
            | c2d3211s + c2d4833s + c2d6433s
            
        x : incA1 x
        
        incA2 :
            | c2d38411s
            | assign shortcut
            
        x : incA2 x
        
        incA3 : 
            | relu
            
        x : incA3 x
        
        
        # nn funziona dobbiamo poter fare dag all interno di altri dag
        # la merge sum non e permessa

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_half_complete_inception(self):
        inc = """
        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        re := Activation 'relu'

        inception_stem : 
                        | c2d32 
                        | c2d32 
                        | c2d32

        x : Input with shape=(32,32,3)

        x : inception_stem x
        
        finish_inception :
                        | AveragePooling2D with pool_size=(4, 4) strides=(1, 1) border_mode='valid' dim_ordering='tf' + Dropout 0.5 + Flatten
        
        model : finish_inception x
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_final(self):
        # x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        # x = Dropout(0.5)(x)
        # x = Flatten()(x)

        inc = """
        finish_inception :
                        | AveragePooling2D with pool_size=(4, 4) strides=(1, 1) border_mode='valid' dim_ordering='tf' + Dropout 0.5 + Flatten
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_dag_more_complex(self):
        inception_layer = """
        seq := Sequential
        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        re := Activation 'relu'

        inception_stem  :
                        | c2d32 
                        | c2d32 
                        | c2d32

        x : Input with shape=(32,32,3)

        model : inception_stem x
        model : inception_stem model
        model : inception_stem model
        
        model : 
                | model 
                | c2d32
        """
        self.mll = MLL(inception_layer)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_dag_complex(self):
        inception_layer = """
        seq := Sequential
        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        re := Activation 'relu'

        inception_stem  :
                        |c2d32 
                        | c2d32 
                        | c2d32
        
        x : Input with shape=(32,32,3)
        
        model : inception_stem x
        """
        self.mll = MLL(inception_layer)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_dag(self):
        inception_layer = """
        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        re := Activation 'relu'

        inception_stem  :
                        | c2d32 
                        | c2d32 + c2d32 
                        | c2d32 + c2d32 + c2d32 
                        | m2d
        """
        self.mll = MLL(inception_layer)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d48 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d192 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d384 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re

        stem : seq + InputLayer (100, 100, 3) + c2d32 + c2d32 + c2d64

        biforcazione1 : stem + m2d
        biforcazione2 : stem + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : concatenate biforcazione1 biforcazione2

        stem : stem + re

        biforcazione1 : stem + c2d32
        biforcazione2 : stem + c2d32 + c2d32
        biforcazione3 : stem + c2d32 + c2d48 + c2d64

        A : concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : concatenate A stem
        A : A + re

        m2d := MaxPooling2D 3, 3 with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : A + m2d
        biforcazione2 : A + c2d384
        biforcazione3 : A + c2d256 + c2d256 + c2d3822

        redA : concatenate biforcazione1 biforcazione2 biforcazione3 

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        #self.mll.execute()

    def test_sk_1(self):
        skmodel4 = """

        criterion have 'gini' or 'entropy'

        rf_clf  : @RandomForestClassifier 10 entropy
        knn_clf : @KNeighborsClassifier 2
        svc_clf : @SVC with C=10000.0
        rg_clf  : @RidgeClassifier 0.1
        dt_clf  : @DecisionTreeClassifier gini
        lr      : @LogisticRegression
        classifier sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        """

        self.mll = MLL(skmodel4)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        train, test = get_data()

        sclf.fit(train, test)

        scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        print(scores.mean(), scores.std())

    def test_simple_net(self):

        simple_net = """
        params_conv2d := (3, 3) with padding='same'
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        ANN := seq

        padding have 'same' or 'valid'

        criterion have 'gini' or 'entropy'

        classifier rf_clf  : @RandomForestClassifier 10 entropy
        knn_clf : @KNeighborsClassifier 2
        svc_clf : @SVC with C=10000.0
        rg_clf  : @RidgeClassifier 0.1
        dt_clf  : @DecisionTreeClassifier gini
        lr      : @LogisticRegression
        sclf : @StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        net : ANN 
        + Conv2D 32 (3, 3) with input_shape=(100, 100, 3)
        + relu
        + flatten
        + Dense 256
        + relu
        + Dropout 0.5
        + Dense 10 with activation='softmax'
        """

        self.mll = MLL(simple_net)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        net = self.mll.last_model()

        # Generate dummy data
        x_train = np.random.random((100, 100, 100, 3))
        y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
        x_test = np.random.random((20, 100, 100, 3))
        y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        net.compile(loss='categorical_crossentropy', optimizer=sgd)

        net.fit(x_train, y_train, batch_size=32, epochs=10)
        score = net.evaluate(x_test, y_test, batch_size=32)

        print(net.summary())

    def test_mll_empty(self):
        MLL("")

    def test_inception_commented(self):
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


        #layer di input riceve in input x
        
        stem : x + c2d32 + c2d32 + c2d64

        biforcazione1 : stem + m2d
        biforcazione2 : stem + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : concatenate biforcazione1 biforcazione2

        stem : stem + re

        #layer A riceve in input x

        biforcazione1 : x + c2d32
        biforcazione2 : x + c2d32 + c2d32
        biforcazione3 : x + c2d32 + c2d48 + c2d64

        A : concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : concatenate A x
        A : A + re

        #layer redA riceve in input x

        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : x + m2d
        biforcazione2 : x + c2d384
        biforcazione3 : x + c2d256 + c2d256 + c2d28422

        redA : concatenate biforcazione1 biforcazione2 biforcazione3 

        #layer B riceve in input x

        #da finire

        """

        self.mll = MLL(inception)
        self.mll.start()
        print(self.mll.get_string())
        # self.mll.execute()

    def test_imports(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211 := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411 := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417 := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471 := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311 := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611 := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411 := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        dense := keras.layers.Dense
        densem := dense 384

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        m :
            | c2d3233 + c2d3233 + c2d6433
            
        x : m x

        stem :
            | m2d3311
            | c2d9633
            | concat
            | c2d6411 + c2d9633
            | c2d6411 + c2d6471 + c2d6417 + c2d9633
            | concat
            | c2d19233
            | m2d3311
            | concat
            | relu + densem
            
        x : stem x
            
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

        print(type(self.mll.last_model()))

    def test_dag_fork_sequential(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(3,3) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(3,3) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4811v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4811s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411sl := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear' 
        
        # Input layer
        
        x : Input with shape = (32,32,3)
        
        stem :
            | c2d3211s
            | c2d3211s + c2d3211s
            | c2d3211s + c2d4811s + c2d6411s
            | concat
            | c2d38411s
        
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

        print(type(self.mll.last_model()))

    def test_auto_import(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(3,3) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(3,3) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4811v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4811s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411sl := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear' 

        # Input layer

        x : Input with shape = (32,32,3)

        stem :
            | c2d3211s
            | c2d3211s + c2d3211s
            | c2d3211s + c2d4811s + c2d6411s
            | concat
            | c2d38411s

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_simpler_auto_import(self):
        skmodel4 = """

        criterion have 'gini' or 'entropy'

        rf_clf  : RandomForestClassifier 10 entropy
        knn_clf : KNeighborsClassifier 2
        svc_clf : SVC with C=10000.0
        rg_clf  : RidgeClassifier 0.1
        dt_clf  : DecisionTreeClassifier gini
        lr      : LogisticRegression
        classifier sclf : StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        """

        self.mll = MLL(skmodel4)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        train, test = get_data()

        sclf.fit(train, test)

        scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        print(scores.mean(), scores.std())

    def test_inception_mod_leave_one_out(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d48 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d192 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d384 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re

        stem : InputLayer (100, 100, 3) 
        
        stem : stem + c2d32 + c2d32 + c2d64

        stem : 
            | right -> | m2d
            | left -> | c2d96
            | loo -> | m2d
            | concat right left
            
        #non puo diventare x l ultima concat

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_mod_double_fork(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d48 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d192 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d384 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re

        stem : InputLayer (100, 100, 3) 

        stem : stem + c2d32 + c2d32 + c2d64

        stem : 
            | right -> | m2d | c2d96 | concat
            | left -> | m2d | c2d96 | concat
            | concat right left

        #le concat nested senza parametri producono le lettere prima della freccia
        #l ultima concat con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_mod_leave_one_out_and_double_fork(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d48 := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d192 := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d384 := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re

        stem : InputLayer (100, 100, 3) 

        stem : stem + c2d32 + c2d32 + c2d64

        stem : 
            | right -> | m2d | c2d96 | concat
            | left -> | m2d | c2d96 | concat
            | loo -> | m2d + c2d96 + c2d96
            | concat a b

        #le concat nested senza parametri producono le lettere prima della freccia
        #l ultima concat con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_commented_commas(self):
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


        #layer di input riceve in input x

        stem : x + c2d32 + c2d32 + c2d64

        biforcazione1 : stem + m2d
        biforcazione2 : stem + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : concatenate biforcazione1 biforcazione2

        stem : stem + re

        #layer A, riceve in input x

        biforcazione1 : x + c2d32
        biforcazione2 : x + c2d32 + c2d32
        biforcazione3 : x + c2d32 + c2d48 + c2d64

        A : concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : concatenate A x
        A : A + re

        #layer redA, riceve in input x

        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : x + m2d
        biforcazione2 : x + c2d384
        biforcazione3 : x + c2d256 + c2d256 + c2d28422

        redA : concatenate biforcazione1 biforcazione2 biforcazione3 

        #layer B, riceve in input x

        #da finire

        """

        self.mll = MLL(inception)
        self.mll.start()
        print(self.mll.get_string())
        # self.mll.execute()

    def test_half_complete_inception_sum(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        stem1 :
            | c2d3233v + c2d3233v + c2d6433s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d9633v

        x : stem2 x

        stem3 :
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v

        x : stem3 x

        stem4 :
            | c2d19233v
            | m2d3311v

        x : stem4 x

        stem5 : 
            | relu

        x : stem5 x

        # layer A

        shortcut : assign x

        incA1 :
            | c2d3211s
            | c2d3211s + c2d3233s
            | c2d3211s + c2d4833s + c2d6433s

        x : incA1 x

        incA2 :
            | c2d38411s
            | assign shortcut
            | sum
            
        #bisogna definire sum

        x : incA2 x
        
        # la parte del concat o sum non e presente nei precedenti tests
        # dovremmo fare una versione di questo test piu corto

        incA3 : 
            | relu

        x : incA3 x


        # nn funziona dobbiamo poter fare dag all interno di altri dag
        # la merge sum non e permessa

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_half_complete_inception_shortened(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        stem1 :
            | c2d3233v + c2d3233v + c2d6433s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d9633v
            | concat
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | concat
            | c2d19233v
            | m2d3311v
            | concat

        x : stem2 x

        stem5 : 
            | relu

        x : stem5 x

        # layer A

        shortcut : assign x

        incA1 :
            | c2d3211s
            | c2d3211s + c2d3233s
            | c2d3211s + c2d4833s + c2d6433s
            | concat
            | c2d38411s
            | assign shortcut
            | sum

        #l ultima concat qui sopra sarebbe una sum
        #bisogna definire sum

        x : incA1 x

        # la parte del concat o sum non e presente nei precedenti tests
        # dovremmo fare una versione di questo test piu corto

        incA2 : 
            | relu

        x : incA2 x


        # nn funziona dobbiamo poter fare dag all interno di altri dag
        # la merge sum non e permessa

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_external_data(self):

        ext = 384

        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d3211v := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3211s := Conv2D 32 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d4833v := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d4833s := Conv2D 48 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6433v := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411v := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417v := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471v := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3311s := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='same' dim_ordering ='tf'
        c2d9611v := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9611s := Conv2D 96 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d9633s := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d19233v := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38433v := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38433s := Conv2D 384 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411v := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d38411s := Conv2D 384 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d38411s_ext := Conv2D @ext (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        stem1 :
            | c2d3233v + c2d3233v + c2d6433s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d9633v
            | concat
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | concat
            | c2d19233v
            | m2d3311v
            | concat

        x : stem2 x

        stem5 : 
            | relu

        x : stem5 x

        # layer A

        shortcut : assign x

        incA1 :
            | c2d3211s
            | c2d3211s + c2d3233s
            | c2d3211s + c2d4833s + c2d6433s
            |concat
            | c2d38411s_ext
            | assign shortcut
            | concat

        #l ultima concat qui sopra sarebbe una sum
        #bisogna definire sum

        x : incA1 x

        # la parte del concat o sum non e presente nei precedenti tests
        # dovremmo fare una versione di questo test piu corto

        incA2 : 
            | relu

        x : incA2 x


        # nn funziona dobbiamo poter fare dag all interno di altri dag
        # la merge sum non e permessa

        """

        self.mll = MLL(inc,locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        #self.mll.image_tree("before")

    def test_external_data_simpler(self):

        ext = 384

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext (1, 1) with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat
            
        x : incA1 x

        """

        self.mll = MLL(inc,locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_func_app(self):
        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()

    def test_wrong_type(self):
        ext = 384

        inc = {}

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_stampa(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        self.mll.print_tree()

    def test_image_tree_after(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        self.mll.image_tree()

    def test_image_tree_before(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        print(self.mll.image_tree("before"))

    def test_get_imports(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        print(self.mll.get_imports())

    def test_get_tree_before_and_list_types(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        assert (isinstance(self.mll.get_tree_before(), Tree))
        list_types(self.mll.get_tree_before().children)

    def test_get_tree_after(self):

        ext = 384

        f = lambda :(1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D @ext @f() with subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : assign x

        incA1 :
            | c2d38411s_ext
            | assign shortcut
            | concat

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()
        assert(isinstance(self.mll.get_tree_after(),Tree))

    def test_inception_mod_double_fork_without_last_concat(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re

        stem : 
            | right -> | m2d | c2d96 | concat
            | left -> | m2d | c2d96 | concat

        #le concat nested senza parametri producono le lettere prima della freccia
        #l ultima concat con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_image_for_thesis(self):
        ext = 384

        inc = """
        conv2d := Conv2D
        seq := Sequential
        relu := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d3233v := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        c2d3233s := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        
        c2d6433s := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6411s := Conv2D 64 (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6417s := Conv2D 64 (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        c2d6471s := Conv2D 64 (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu
        
        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        
        c2d9633v := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
        
        c2d19233s := Conv2D 192 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + relu

        # Input layer

        x : Input with shape = (32,32,3)

        # Layer stem di entrata dell input

        stem1 :
            | c2d3233v + c2d3233v + c2d6433s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d9633v
            | concat
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | concat

        x : stem2 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

    def test_inception_mod_double_fork_without_last_concat_for_thesis(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        m2d := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re

        stem : 
            | right -> | m2d | c2d96 | concat
            | left -> | m2d | c2d96 | concat

        #le concat nested senza parametri producono le lettere prima della freccia
        #l ultima concat con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

    def test_keras_layer_import_lower(self):
        print("sum" in keras.backend.__dict__.keys())

        a = []

        for i in keras.backend.__dict__.keys():
            if str(i).islower() and "__" not in i:
                a+=[i]

        print(a)