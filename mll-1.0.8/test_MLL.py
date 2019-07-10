from unittest import TestCase

from lark import Tree
from sklearn import model_selection
from termcolor import cprint

from mll.mlltranspiler import MLL

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import warnings

from mll.utils import get_keras_layers

warnings.filterwarnings("ignore")

import numpy as np
import keras
from keras.optimizers import SGD


def get_data():
    iris_dataset = pd.read_csv("iris.csv")
    train, test = iris_dataset.iloc[:, 0:4], iris_dataset.iloc[:, 4]

    encoder_object = LabelEncoder()
    test = encoder_object.fit_transform(test)
    print(test)

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
        
        stem :
            | c2d3233 + c2d3233 + c2d6433
            
        x : stem x

        stem2 :
            | m2d3311
            | c2d9633
            | Concatenate
            | c2d6411 + c2d9633
            | c2d6411 + c2d6471 + c2d6417 + c2d9633
            | Concatenate
            | c2d19233
            | m2d3311
            | Concatenate
            | relu
            
        x : stem2 x

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

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : Concatenate biforcazione1 biforcazione2

        stem : stem + re

        biforcazione1 : stem + c2d32
        biforcazione2 : stem + c2d32 + c2d32
        biforcazione3 : stem + c2d32 + c2d48 + c2d64

        A : Concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : Concatenate A stem
        A : A + re

        m2d := MaxPooling2D 3, 3 with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : A + m2d
        biforcazione2 : A + c2d384
        biforcazione3 : A + c2d256 + c2d256 + c2d3822

        redA : Concatenate biforcazione1 biforcazione2 biforcazione3 

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        #self.mll.execute()

    def test_sk_1(self):
        skmodel4 = """

        criterion IS 'gini' or 'entropy'

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

        padding IS 'same' or 'valid'

        criterion IS 'gini' or 'entropy'

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

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : Concatenate biforcazione1 biforcazione2

        stem : stem + re

        #layer A riceve in input x

        biforcazione1 : x + c2d32
        biforcazione2 : x + c2d32 + c2d32
        biforcazione3 : x + c2d32 + c2d48 + c2d64

        A : Concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : Concatenate A x
        A : A + re

        #layer redA riceve in input x

        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : x + m2d
        biforcazione2 : x + c2d384
        biforcazione3 : x + c2d256 + c2d256 + c2d28422

        redA : Concatenate biforcazione1 biforcazione2 biforcazione3 

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
            | Concatenate
            | c2d6411 + c2d9633
            | c2d6411 + c2d6471 + c2d6417 + c2d9633
            | Concatenate
            | c2d19233
            | m2d3311
            | Concatenate
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
        c2d38411sl := Conv2D 384 (1, 1) with subsample=(1,1) 
        
        # Input layer
        
        x : Input with shape = (32,32,3)
        
        stem :
            | c2d3211s
            | c2d3211s + c2d3211s
            | c2d3211s + c2d4811s + c2d6411s
            | Concatenate
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
            | Concatenate
            | c2d38411s

        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_simpler_auto_import(self):
        skmodel4 = """

        criterion  'gini' or 'entropy'

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

        x : Input with shape = (32,32,3)
        
        stem : 
            | c2d32 + c2d32 + c2d64
            
        x : stem x

        stem2 : 
            | right -> | c2d96 | m2d | Concatenate
            | left -> | c2d96 | m2d | Concatenate
            | loo -> | m2d
            | Concatenate | right | left
            
        x : stem2 x
            
        #non puo diventare x l ultima Concatenate

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

        x : Input with shape = (32,32,3)

        stem : 
            | c2d32 + c2d32 + c2d64
            
        x : stem x

        stem2 : 
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate
            | Concatenate | right | left
            
        x : stem2 x

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

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

        x : Input with shape = (32,32,3)

        stem : 
            | c2d32 + c2d32 + c2d64

        x : stem x

        stem2 : 
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate
            | loo -> | m2d + c2d96 + c2d96
            | Concatenate | right | left

        x : stem2 x

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_inception_mod_double_fork_concat_for_thesis(self):
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

        x : Input with shape = (32,32,3)

        stem : 
            | c2d32 + c2d32 + c2d64

        x : stem x

        stem2 : 
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate
            | Concatenate | right | left

        x : stem2 x

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

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

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d64 + c2d96
        biforcazione2 : stem + c2d64 + c2d64 + c2d64 + c2d96

        stem : Concatenate biforcazione1 biforcazione2

        biforcazione1 : stem + c2d192
        biforcazione2 : stem + m2d

        stem : Concatenate biforcazione1 biforcazione2

        stem : stem + re

        #layer A, riceve in input x

        biforcazione1 : x + c2d32
        biforcazione2 : x + c2d32 + c2d32
        biforcazione3 : x + c2d32 + c2d48 + c2d64

        A : Concatenate biforcazione1 biforcazione2 biforcazione3

        A : A + c2d384
        A : Concatenate A x
        A : A + re

        #layer redA, riceve in input x

        m2d := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        c2d384 := Conv2D 384 3 3 with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d256 := Conv2D 384 1 1 with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re
        c2d3822 := Conv2D 384 1 1 with subsample=(2,2) init='he_normal' border_mode='same' dim_ordering='tf' + re

        biforcazione1 : x + m2d
        biforcazione2 : x + c2d384
        biforcazione3 : x + c2d256 + c2d256 + c2d28422

        redA : Concatenate biforcazione1 biforcazione2 biforcazione3 

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
        
        # la parte del Concatenate o sum non e presente nei precedenti tests
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
            | Concatenate
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | Concatenate
            | c2d19233v
            | m2d3311v
            | Concatenate

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
            | Concatenate
            | c2d38411s
            | assign shortcut
            | sum

        #l ultima Concatenate qui sopra sarebbe una sum
        #bisogna definire sum

        x : incA1 x

        # la parte del Concatenate o sum non e presente nei precedenti tests
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
            | Concatenate
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | Concatenate
            | c2d19233v
            | m2d3311v
            | Concatenate

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
            |Concatenate
            | c2d38411s_ext
            | assign shortcut
            | Concatenate

        #l ultima Concatenate qui sopra sarebbe una sum
        #bisogna definire sum

        x : incA1 x

        # la parte del Concatenate o sum non e presente nei precedenti tests
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
            | Concatenate
            
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
            | Concatenate

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
            | Concatenate

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
            | Concatenate

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
            | Concatenate

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
            | Concatenate

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
            | Concatenate

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
            | Concatenate

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
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

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
            | Concatenate
            | c2d6411s + c2d9633v
            | c2d6411s + c2d6471s + c2d6417s + c2d9633v
            | Concatenate

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

        m2d := MaxPooling2D (3, 3) with dim_ordering='tf' strides=(1, 1) border_mode='valid'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re

        stem : 
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

    def test_inception_mod_double_fork_without_last_concat_with_dag_parmac(self):
        inception_uncomm = """
        conv2d := Conv2D
        seq := Sequential
        re := Activation 'relu'
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        dim_ordering IS 'tf'

        m2d := MaxPooling2D (3, 3) tf with strides=(1, 1) border_mode='valid'
        c2d96 := Conv2D 96 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re

        stem : 
            | right -> | m2d | c2d96 | Concatenate
            | left -> | m2d | c2d96 | Concatenate

        #le Concatenate nested senza parametri producono le lettere prima della freccia
        #l ultima Concatenate con paramteri produce x

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

    def test_image_for_thesis_simple_concat(self):
        ext = 384

        inc = """
        conv2d := Conv2D
        relu := Activation 'relu'

        c2d3233 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' dim_ordering='tf' + relu
        c2d6433 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' dim_ordering='tf' + relu
        
        stem:
            | c2d3233 + c2d3233 + c2d6433
            | c2d3233 + c2d3233 + c2d6433
            | Concatenate

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

    def test_image_for_thesis_simple(self):
        ext = 384

        inc = """
        conv2d := Conv2D
        relu := Activation 'relu'

        c2d3233 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' dim_ordering='tf' + relu
        c2d6433 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' dim_ordering='tf' + relu

        stem :
              | c2d3233 + c2d3233 + c2d6433

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        self.mll.image_tree("before")

    def test_entire_inception_with_function_bigger(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d323311v := Conv2D @fant(32) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d323311s := Conv2D @fant(32) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d321111v := Conv2D @fant(32) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d321111s := Conv2D @fant(32) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d483311v := Conv2D @fant(48) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d483311s := Conv2D @fant(48) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d643311v := Conv2D @fant(64) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d643311s := Conv2D @fant(64) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641111v := Conv2D @fant(64) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641111s := Conv2D @fant(64) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641711v := Conv2D @fant(64) (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641711s := Conv2D @fant(64) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d647111v := Conv2D @fant(64) (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d647111s := Conv2D @fant(64) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d961111v := Conv2D @fant(96) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d961111s := Conv2D @fant(96) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d963311v := Conv2D @fant(96) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d963311s := Conv2D @fant(96) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1281111s := Conv2D @fant(128) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1601711s := Conv2D @fant(160) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1607111s := Conv2D @fant(160) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1923311v := Conv2D @fant(192) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d1923311s := Conv2D @fant(192) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1921111s := Conv2D @fant(192) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1927111s := Conv2D @fant(192) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2241311s := Conv2D @fant(224) (1, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2561111s := Conv2D @fant(256) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563311s := Conv2D @fant(256) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563111s := Conv2D @fant(256) (3, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2883322v := Conv2D @fant(288) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d2883311s:= Conv2D @fant(288) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d3203322v := Conv2D @fant(320) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d3843311v := Conv2D @fant(384) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3843311s := Conv2D @fant(384) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d3841111v := Conv2D @fant(384) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3841111slin := Conv2D @fant(384) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d3843322v := Conv2D @fant(384) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d11541111slin := Conv2D @fant(1154) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d20481111slin := Conv2D @fant(2048) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | relu

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | relu

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | relu

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | relu

        x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        with open('my-inception-report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # In[11]:

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # In[12]:

        batch_size = 128
        nb_epoch = 10
        data_augmentation = False

        # Model saving callback
        # checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

        # if not data_augmentation:
        #     print('Not using data augmentation.')
        #     history = model.fit(x_train, y_train,
        #                         batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
        #                         validation_data=(x_test, y_test), shuffle=True,
        #                         callbacks=[])
        # else:
        #     print('Using real-time data augmentation.')
        #
        #     # realtime data augmentation
        #     datagen_train = ImageDataGenerator(
        #         featurewise_center=False,
        #         samplewise_center=False,
        #         featurewise_std_normalization=False,
        #         samplewise_std_normalization=False,
        #         zca_whitening=False,
        #         rotation_range=0,
        #         width_shift_range=0.125,
        #         height_shift_range=0.125,
        #         horizontal_flip=True,
        #         vertical_flip=False)
        #     datagen_train.fit(x_train)
        #
        #     # fit the model on the batches generated by datagen.flow()
        #     history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
        #                                   samples_per_epoch=x_train.shape[0],
        #                                   nb_epoch=nb_epoch, verbose=1,
        #                                   validation_data=(x_test, y_test),
        #                                   callbacks=[])

    def test_entire_inception_with_function_bigger_aug_with_parmac(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        re := Activation 'relu'
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        
        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf' or 'th'
        activation IS 'relu' or 'linear'
        
        kkk := he_normal, valid, tf, relu

        c2d323311v := Conv2D @fant(32) (3, 3) kkk with subsample=(1,1)
        c2d323311s := Conv2D @fant(32) (3, 3) he_normal same tf relu with subsample=(1,1)
        c2d321111v := Conv2D @fant(32) (1, 1) he_normal valid tf relu with subsample=(1,1)
        c2d321111s := Conv2D @fant(32) (1, 1) he_normal same tf relu with subsample=(1,1)


        c2d483311v := Conv2D @fant(48) (3, 3) he_normal valid tf relu with subsample=(1,1)
        c2d483311s := Conv2D @fant(48) (3, 3) he_normal same tf relu with subsample=(1,1)


        c2d643311v := Conv2D @fant(64) (3, 3) he_normal valid tf relu with subsample=(1,1)
        c2d643311s := Conv2D @fant(64) (3, 3) he_normal same tf relu with subsample=(1,1)
        c2d641111v := Conv2D @fant(64) (1, 1) he_normal valid tf relu with subsample=(1,1)
        c2d641111s := Conv2D @fant(64) (1, 1) he_normal same tf relu with subsample=(1,1)
        c2d641711v := Conv2D @fant(64) (1, 7) he_normal valid tf relu with subsample=(1,1)
        c2d641711s := Conv2D @fant(64) (1, 7) he_normal same tf relu with subsample=(1,1)
        c2d647111v := Conv2D @fant(64) (7, 1) he_normal valid tf relu with subsample=(1,1)
        c2d647111s := Conv2D @fant(64) (7, 1) he_normal same tf relu with subsample=(1,1)


        c2d961111v := Conv2D @fant(96) (1, 1) he_normal valid tf relu with subsample=(1,1)
        c2d961111s := Conv2D @fant(96) (1, 1) he_normal same tf relu with subsample=(1,1)
        c2d963311v := Conv2D @fant(96) (3, 3) he_normal valid tf relu with subsample=(1,1)
        c2d963311s := Conv2D @fant(96) (3, 3) he_normal same tf relu with subsample=(1,1)


        c2d1281111s := Conv2D @fant(128) (1, 1) he_normal same tf relu with subsample=(1,1)


        c2d1601711s := Conv2D @fant(160) (1, 7) he_normal same tf relu with subsample=(1,1)
        c2d1607111s := Conv2D @fant(160) (7, 1) he_normal same tf relu with subsample=(1,1)


        c2d1923311v := Conv2D @fant(192) (3, 3) he_normal valid tf relu with subsample=(1,1)
        c2d1923311s := Conv2D @fant(192) (3, 3) he_normal same tf relu with subsample=(1,1)
        c2d1921111s := Conv2D @fant(192) (1, 1) he_normal same tf relu with subsample=(1,1)
        c2d1927111s := Conv2D @fant(192) (7, 1) he_normal same tf relu with subsample=(1,1)


        c2d2241311s := Conv2D @fant(224) (1, 3) he_normal same tf relu with subsample=(1,1)


        c2d2561111s := Conv2D @fant(256) (1, 1) he_normal same tf relu with subsample=(1,1)
        c2d2563311s := Conv2D @fant(256) (3, 3) he_normal same tf relu with subsample=(1,1)
        c2d2563111s := Conv2D @fant(256) (3, 1) he_normal same tf relu with subsample=(1,1)


        c2d2883322v := Conv2D @fant(288) (3, 3) he_normal valid tf relu with subsample=(2,2)
        c2d2883311s:= Conv2D @fant(288) (3, 3) he_normal same tf relu with subsample=(1,1)


        c2d3203322v := Conv2D @fant(320) (3, 3) he_normal valid tf relu with subsample=(2,2)


        c2d3843311v := Conv2D @fant(384) (3, 3) he_normal valid tf relu with subsample=(1,1)
        c2d3843311s := Conv2D @fant(384) (3, 3) with subsample=(1,1)
        c2d3841111v := Conv2D @fant(384) (1, 1) he_normal valid tf relu with subsample=(1,1)
        c2d3841111slin := Conv2D @fant(384) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d3843322v := Conv2D @fant(384) (3, 3) he_normal valid tf relu with subsample=(2,2)


        c2d11541111slin := Conv2D @fant(1154) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d20481111slin := Conv2D @fant(2048) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | re

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | re

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | re

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | re

        x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        # with open('my-inception-report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        # if not data_augmentation:
        #     print('Not using data augmentation.')
        #     history = model.fit(x_train, y_train,
        #                         batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
        #                         validation_data=(x_test, y_test), shuffle=True,
        #                         callbacks=[])
        # else:
        #     print('Using real-time data augmentation.')
        #
        #     # realtime data augmentation
        #     datagen_train = ImageDataGenerator(
        #         featurewise_center=False,
        #         samplewise_center=False,
        #         featurewise_std_normalization=False,
        #         samplewise_std_normalization=False,
        #         zca_whitening=False,
        #         rotation_range=0,
        #         width_shift_range=0.125,
        #         height_shift_range=0.125,
        #         horizontal_flip=True,
        #         vertical_flip=False)
        #     datagen_train.fit(x_train)
        #
        #     # fit the model on the batches generated by datagen.flow()
        #     history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
        #                                   samples_per_epoch=x_train.shape[0],
        #                                   nb_epoch=nb_epoch, verbose=1,
        #                                   validation_data=(x_test, y_test),
        #                                   callbacks=[])

    def test_entire_inception_with_function_bigger_aug_with_parmac_mini(self):
        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        re := Activation 'relu'
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf' or 'th'
        activation IS 'relu' or 'linear'

        mmm := he_normal, valid, tf, relu

        c2d323311v := Conv2D 32 (3, 3) mmm with subsample=(1,1)

        stem1 :
            | c2d323311v + c2d323311v

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.image_tree("before")
        self.mll.execute()

    def test_inception_mod_double_fork_for_thesis(self):
        inception_uncomm = """
        re := Activation 'relu'

        c2d32 := Conv2D 32 (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + re
        c2d64 := Conv2D 64 (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' + re

        x : Input with shape = (32,32,3)

        stem : 
            | c2d32 + c2d32 + c2d64

        x : stem x

        """
        self.mll = MLL(inception_uncomm)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_entire_inception_with_function_bigger_aug_shortened(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        re := Activation 'relu'

        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf'
        activation IS 'relu' or 'linear'

        val_rel := he_normal, valid, tf, relu
        val_rel := he_normal, valid, tf, linear
        sam_rel := he_normal, same, tf, relu
        sam_lin := he_normal, same, tf, linear
        m2dv := valid, tf
        m2ds := same, tf

        c2d323311v := Conv2D @fant(32) (3, 3) val_rel with subsample=(1,1) 
        c2d323311s := Conv2D @fant(32) (3, 3) sam_rel with subsample=(1,1)
        c2d321111v := Conv2D @fant(32) (1, 1) val_rel with subsample=(1,1) 
        c2d321111s := Conv2D @fant(32) (1, 1) sam_rel with subsample=(1,1)


        c2d483311v := Conv2D @fant(48) (3, 3) val_rel with subsample=(1,1) 
        c2d483311s := Conv2D @fant(48) (3, 3) sam_rel with subsample=(1,1)


        c2d643311v := Conv2D @fant(64) (3, 3) val_rel with subsample=(1,1) 
        c2d643311s := Conv2D @fant(64) (3, 3) sam_rel with subsample=(1,1)
        c2d641111v := Conv2D @fant(64) (1, 1) val_rel with subsample=(1,1) 
        c2d641111s := Conv2D @fant(64) (1, 1) sam_rel with subsample=(1,1)
        c2d641711v := Conv2D @fant(64) (1, 7) val_rel with subsample=(1,1) 
        c2d641711s := Conv2D @fant(64) (1, 7) sam_rel with subsample=(1,1)
        c2d647111v := Conv2D @fant(64) (7, 1) val_rel with subsample=(1,1) 
        c2d647111s := Conv2D @fant(64) (7, 1) sam_rel with subsample=(1,1)


        c2d961111v := Conv2D @fant(96) (1, 1) val_rel with subsample=(1,1) 
        c2d961111s := Conv2D @fant(96) (1, 1) sam_rel with subsample=(1,1)
        c2d963311v := Conv2D @fant(96) (3, 3) val_rel with subsample=(1,1) 
        c2d963311s := Conv2D @fant(96) (3, 3) sam_rel with subsample=(1,1)


        c2d1281111s := Conv2D @fant(128) (1, 1) sam_rel with subsample=(1,1)


        c2d1601711s := Conv2D @fant(160) (1, 7) sam_rel with subsample=(1,1)
        c2d1607111s := Conv2D @fant(160) (7, 1) sam_rel with subsample=(1,1)


        c2d1923311v := Conv2D @fant(192) (3, 3) val_rel with subsample=(1,1) 
        c2d1923311s := Conv2D @fant(192) (3, 3) sam_rel with subsample=(1,1)
        c2d1921111s := Conv2D @fant(192) (1, 1) sam_rel with subsample=(1,1)
        c2d1927111s := Conv2D @fant(192) (7, 1) sam_rel with subsample=(1,1)


        c2d2241311s := Conv2D @fant(224) (1, 3) sam_rel with subsample=(1,1)


        c2d2561111s := Conv2D @fant(256) (1, 1) sam_rel with subsample=(1,1)
        c2d2563311s := Conv2D @fant(256) (3, 3) sam_rel with subsample=(1,1)
        c2d2563111s := Conv2D @fant(256) (3, 1) sam_rel with subsample=(1,1)


        c2d2883322v := Conv2D @fant(288) (3, 3) val_rel with subsample=(2,2) 
        c2d2883311s:= Conv2D @fant(288) (3, 3) sam_rel with subsample=(1,1)


        c2d3203322v := Conv2D @fant(320) (3, 3) val_rel with subsample=(2,2) 


        c2d3843311v := Conv2D @fant(384) (3, 3) val_rel with subsample=(1,1) 
        c2d3843311s := Conv2D @fant(384) (3, 3) sam_rel with subsample=(1,1)
        c2d3841111v := Conv2D @fant(384) (1, 1) val_rel with subsample=(1,1) 
        c2d3841111slin := Conv2D @fant(384) (1, 1) sam_lin with subsample=(1,1)


        c2d3843322v := Conv2D @fant(384) (3, 3) val_rel with subsample=(2,2) 


        c2d11541111slin := Conv2D @fant(1154) (1, 1) sam_lin with subsample=(1,1)


        c2d20481111slin := Conv2D @fant(2048) (1, 1) sam_lin with subsample=(1,1)


        m2d3311v := MaxPooling2D (3, 3) m2dv with strides=(1, 1)
        m2d3322s := MaxPooling2D (3, 3) m2ds with strides=(2, 2)
        m2d3322v := MaxPooling2D (3, 3) m2dv with strides=(2, 2)
        m2d3322v := MaxPooling2D (3, 3) m2dv with strides=(2, 2)


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | re

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | re

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | re

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | re

        x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        # with open('my-inception-report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        run = False

        if run:
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

    def fant(x):
        cprint("sono nella fant esterna","blue")
        return x

    def test_entire_inception_with_function_bigger_aug_shortened_external_fant(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        re := Activation 'relu'

        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf'
        activation IS 'relu' or 'linear'

        val_rel := he_normal, valid, tf, relu
        val_rel := he_normal, valid, tf, linear
        sam_rel := he_normal, same, tf, relu
        sam_lin := he_normal, same, tf, linear

        c2d323311v := Conv2D @fant(32) (3, 3) val_rel with subsample=(1,1) 
        c2d323311s := Conv2D @fant(32) (3, 3) sam_rel with subsample=(1,1)
        c2d321111v := Conv2D @fant(32) (1, 1) val_rel with subsample=(1,1) 
        c2d321111s := Conv2D @fant(32) (1, 1) sam_rel with subsample=(1,1)


        c2d483311v := Conv2D @fant(48) (3, 3) val_rel with subsample=(1,1) 
        c2d483311s := Conv2D @fant(48) (3, 3) sam_rel with subsample=(1,1)


        c2d643311v := Conv2D @fant(64) (3, 3) val_rel with subsample=(1,1) 
        c2d643311s := Conv2D @fant(64) (3, 3) sam_rel with subsample=(1,1)
        c2d641111v := Conv2D @fant(64) (1, 1) val_rel with subsample=(1,1) 
        c2d641111s := Conv2D @fant(64) (1, 1) sam_rel with subsample=(1,1)
        c2d641711v := Conv2D @fant(64) (1, 7) val_rel with subsample=(1,1) 
        c2d641711s := Conv2D @fant(64) (1, 7) sam_rel with subsample=(1,1)
        c2d647111v := Conv2D @fant(64) (7, 1) val_rel with subsample=(1,1) 
        c2d647111s := Conv2D @fant(64) (7, 1) sam_rel with subsample=(1,1)


        c2d961111v := Conv2D @fant(96) (1, 1) val_rel with subsample=(1,1) 
        c2d961111s := Conv2D @fant(96) (1, 1) sam_rel with subsample=(1,1)
        c2d963311v := Conv2D @fant(96) (3, 3) val_rel with subsample=(1,1) 
        c2d963311s := Conv2D @fant(96) (3, 3) sam_rel with subsample=(1,1)


        c2d1281111s := Conv2D @fant(128) (1, 1) sam_rel with subsample=(1,1)


        c2d1601711s := Conv2D @fant(160) (1, 7) sam_rel with subsample=(1,1)
        c2d1607111s := Conv2D @fant(160) (7, 1) sam_rel with subsample=(1,1)


        c2d1923311v := Conv2D @fant(192) (3, 3) val_rel with subsample=(1,1) 
        c2d1923311s := Conv2D @fant(192) (3, 3) sam_rel with subsample=(1,1)
        c2d1921111s := Conv2D @fant(192) (1, 1) sam_rel with subsample=(1,1)
        c2d1927111s := Conv2D @fant(192) (7, 1) sam_rel with subsample=(1,1)


        c2d2241311s := Conv2D @fant(224) (1, 3) sam_rel with subsample=(1,1)


        c2d2561111s := Conv2D @fant(256) (1, 1) sam_rel with subsample=(1,1)
        c2d2563311s := Conv2D @fant(256) (3, 3) sam_rel with subsample=(1,1)
        c2d2563111s := Conv2D @fant(256) (3, 1) sam_rel with subsample=(1,1)


        c2d2883322v := Conv2D @fant(288) (3, 3) val_rel with subsample=(2,2) 
        c2d2883311s:= Conv2D @fant(288) (3, 3) sam_rel with subsample=(1,1)


        c2d3203322v := Conv2D @fant(320) (3, 3) val_rel with subsample=(2,2) 


        c2d3843311v := Conv2D @fant(384) (3, 3) val_rel with subsample=(1,1) 
        c2d3843311s := Conv2D @fant(384) (3, 3) sam_rel with subsample=(1,1)
        c2d3841111v := Conv2D @fant(384) (1, 1) val_rel with subsample=(1,1) 
        c2d3841111slin := Conv2D @fant(384) (1, 1) sam_lin with subsample=(1,1)


        c2d3843322v := Conv2D @fant(384) (3, 3) val_rel with subsample=(2,2) 


        c2d11541111slin := Conv2D @fant(1154) (1, 1) sam_lin with subsample=(1,1)


        c2d20481111slin := Conv2D @fant(2048) (1, 1) sam_lin with subsample=(1,1)


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | re

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | re

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | re

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | re

        x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        # with open('my-inception-report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        run = False

        if run:
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

    def test_entire_inception_with_function_bigger_aug_shortened_no_locals(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        re := Activation 'relu'

        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf'
        activation IS 'relu' or 'linear'

        val_rel := he_normal, valid, tf, relu
        val_rel := he_normal, valid, tf, linear
        sam_rel := he_normal, same, tf, relu
        sam_lin := he_normal, same, tf, linear

        c2d323311v := Conv2D @fant(32) (3, 3) val_rel with subsample=(1,1) 
        c2d323311s := Conv2D @fant(32) (3, 3) sam_rel with subsample=(1,1)
        c2d321111v := Conv2D @fant(32) (1, 1) val_rel with subsample=(1,1) 
        c2d321111s := Conv2D @fant(32) (1, 1) sam_rel with subsample=(1,1)


        c2d483311v := Conv2D @fant(48) (3, 3) val_rel with subsample=(1,1) 
        c2d483311s := Conv2D @fant(48) (3, 3) sam_rel with subsample=(1,1)


        c2d643311v := Conv2D @fant(64) (3, 3) val_rel with subsample=(1,1) 
        c2d643311s := Conv2D @fant(64) (3, 3) sam_rel with subsample=(1,1)
        c2d641111v := Conv2D @fant(64) (1, 1) val_rel with subsample=(1,1) 
        c2d641111s := Conv2D @fant(64) (1, 1) sam_rel with subsample=(1,1)
        c2d641711v := Conv2D @fant(64) (1, 7) val_rel with subsample=(1,1) 
        c2d641711s := Conv2D @fant(64) (1, 7) sam_rel with subsample=(1,1)
        c2d647111v := Conv2D @fant(64) (7, 1) val_rel with subsample=(1,1) 
        c2d647111s := Conv2D @fant(64) (7, 1) sam_rel with subsample=(1,1)


        c2d961111v := Conv2D @fant(96) (1, 1) val_rel with subsample=(1,1) 
        c2d961111s := Conv2D @fant(96) (1, 1) sam_rel with subsample=(1,1)
        c2d963311v := Conv2D @fant(96) (3, 3) val_rel with subsample=(1,1) 
        c2d963311s := Conv2D @fant(96) (3, 3) sam_rel with subsample=(1,1)


        c2d1281111s := Conv2D @fant(128) (1, 1) sam_rel with subsample=(1,1)


        c2d1601711s := Conv2D @fant(160) (1, 7) sam_rel with subsample=(1,1)
        c2d1607111s := Conv2D @fant(160) (7, 1) sam_rel with subsample=(1,1)


        c2d1923311v := Conv2D @fant(192) (3, 3) val_rel with subsample=(1,1) 
        c2d1923311s := Conv2D @fant(192) (3, 3) sam_rel with subsample=(1,1)
        c2d1921111s := Conv2D @fant(192) (1, 1) sam_rel with subsample=(1,1)
        c2d1927111s := Conv2D @fant(192) (7, 1) sam_rel with subsample=(1,1)


        c2d2241311s := Conv2D @fant(224) (1, 3) sam_rel with subsample=(1,1)


        c2d2561111s := Conv2D @fant(256) (1, 1) sam_rel with subsample=(1,1)
        c2d2563311s := Conv2D @fant(256) (3, 3) sam_rel with subsample=(1,1)
        c2d2563111s := Conv2D @fant(256) (3, 1) sam_rel with subsample=(1,1)


        c2d2883322v := Conv2D @fant(288) (3, 3) val_rel with subsample=(2,2) 
        c2d2883311s:= Conv2D @fant(288) (3, 3) sam_rel with subsample=(1,1)


        c2d3203322v := Conv2D @fant(320) (3, 3) val_rel with subsample=(2,2) 


        c2d3843311v := Conv2D @fant(384) (3, 3) val_rel with subsample=(1,1) 
        c2d3843311s := Conv2D @fant(384) (3, 3) sam_rel with subsample=(1,1)
        c2d3841111v := Conv2D @fant(384) (1, 1) val_rel with subsample=(1,1) 
        c2d3841111slin := Conv2D @fant(384) (1, 1) sam_lin with subsample=(1,1)


        c2d3843322v := Conv2D @fant(384) (3, 3) val_rel with subsample=(2,2) 


        c2d11541111slin := Conv2D @fant(1154) (1, 1) sam_lin with subsample=(1,1)


        c2d20481111slin := Conv2D @fant(2048) (1, 1) sam_lin with subsample=(1,1)


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | re

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | re

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | re

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | re

        x : incC x

        """

        try:

            self.mll = MLL(inc)
            self.mll.start()
            print(self.mll.get_string())
            self.mll.execute()
            x = self.mll.last_model()

            x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            predictions = Dense(nb_classes, activation='softmax')(x)

            model = Model(input=inputs, output=predictions)

            # In[10]:

            model.summary()

            # with open('my-inception-report.txt', 'w') as fh:
            #     # Pass the file handle in as a lambda function to make it callable
            #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

            run = False

            if run:
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
        except:
            cprint("---------------------------------------------------","green")
            cprint("il test è correttamente fallito perchè è necessario includere il locals() se si vuole passare una funzione definita in questo scope","green")
            cprint("---------------------------------------------------", "green")

    def test_entire_inception_with_function_bigger_aug_shortened_no_locals_with_reg_and_class(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'
        re := Activation 'relu'

        init IS 'he_normal'
        border_mode IS 'valid' or 'same'
        dim_ordering IS 'tf'
        activation IS 'relu' or 'linear'

        val_rel := he_normal, valid, tf, relu
        val_rel := he_normal, valid, tf, linear
        sam_rel := he_normal, same, tf, relu
        sam_lin := he_normal, same, tf, linear

        c2d323311v := Conv2D @fant(32) (3, 3) val_rel with subsample=(1,1) 
        c2d323311s := Conv2D @fant(32) (3, 3) sam_rel with subsample=(1,1)
        c2d321111v := Conv2D @fant(32) (1, 1) val_rel with subsample=(1,1) 
        c2d321111s := Conv2D @fant(32) (1, 1) sam_rel with subsample=(1,1)


        c2d483311v := Conv2D @fant(48) (3, 3) val_rel with subsample=(1,1) 
        c2d483311s := Conv2D @fant(48) (3, 3) sam_rel with subsample=(1,1)


        c2d643311v := Conv2D @fant(64) (3, 3) val_rel with subsample=(1,1) 
        c2d643311s := Conv2D @fant(64) (3, 3) sam_rel with subsample=(1,1)
        c2d641111v := Conv2D @fant(64) (1, 1) val_rel with subsample=(1,1) 
        c2d641111s := Conv2D @fant(64) (1, 1) sam_rel with subsample=(1,1)
        c2d641711v := Conv2D @fant(64) (1, 7) val_rel with subsample=(1,1) 
        c2d641711s := Conv2D @fant(64) (1, 7) sam_rel with subsample=(1,1)
        c2d647111v := Conv2D @fant(64) (7, 1) val_rel with subsample=(1,1) 
        c2d647111s := Conv2D @fant(64) (7, 1) sam_rel with subsample=(1,1)


        c2d961111v := Conv2D @fant(96) (1, 1) val_rel with subsample=(1,1) 
        c2d961111s := Conv2D @fant(96) (1, 1) sam_rel with subsample=(1,1)
        c2d963311v := Conv2D @fant(96) (3, 3) val_rel with subsample=(1,1) 
        c2d963311s := Conv2D @fant(96) (3, 3) sam_rel with subsample=(1,1)


        c2d1281111s := Conv2D @fant(128) (1, 1) sam_rel with subsample=(1,1)


        c2d1601711s := Conv2D @fant(160) (1, 7) sam_rel with subsample=(1,1)
        c2d1607111s := Conv2D @fant(160) (7, 1) sam_rel with subsample=(1,1)


        c2d1923311v := Conv2D @fant(192) (3, 3) val_rel with subsample=(1,1) 
        c2d1923311s := Conv2D @fant(192) (3, 3) sam_rel with subsample=(1,1)
        c2d1921111s := Conv2D @fant(192) (1, 1) sam_rel with subsample=(1,1)
        c2d1927111s := Conv2D @fant(192) (7, 1) sam_rel with subsample=(1,1)


        c2d2241311s := Conv2D @fant(224) (1, 3) sam_rel with subsample=(1,1)


        c2d2561111s := Conv2D @fant(256) (1, 1) sam_rel with subsample=(1,1)
        c2d2563311s := Conv2D @fant(256) (3, 3) sam_rel with subsample=(1,1)
        c2d2563111s := Conv2D @fant(256) (3, 1) sam_rel with subsample=(1,1)


        c2d2883322v := Conv2D @fant(288) (3, 3) val_rel with subsample=(2,2) 
        c2d2883311s:= Conv2D @fant(288) (3, 3) sam_rel with subsample=(1,1)


        c2d3203322v := Conv2D @fant(320) (3, 3) val_rel with subsample=(2,2) 


        c2d3843311v := Conv2D @fant(384) (3, 3) val_rel with subsample=(1,1) 
        c2d3843311s := Conv2D @fant(384) (3, 3) sam_rel with subsample=(1,1)
        c2d3841111v := Conv2D @fant(384) (1, 1) val_rel with subsample=(1,1) 
        c2d3841111slin := Conv2D @fant(384) (1, 1) sam_lin with subsample=(1,1)


        c2d3843322v := Conv2D @fant(384) (3, 3) val_rel with subsample=(2,2) 


        c2d11541111slin := Conv2D @fant(1154) (1, 1) sam_lin with subsample=(1,1)


        c2d20481111slin := Conv2D @fant(2048) (1, 1) sam_lin with subsample=(1,1)


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | re

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | re

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | re

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | re

        regressor x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        print(self.mll.model_type)

        # with open('my-inception-report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        run = False

        if run:
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


    # The 2 tests here below request 1hr each and have to give same result to prove correctness of mll, run them wisely

    def test_real_inception_aug(self):
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
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        def inception_resnet_v2_stem(x):
            # in original inception-resnet-v2, conv stride is 2
            x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(x)
            x = Convolution2D(32 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(x)
            x = Convolution2D(64 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            # in original inception-resnet-v2, stride is 2
            a = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
            # in original inception-resnet-v2, conv stride is 2
            b = Convolution2D(96 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(x)
            x = merge([a, b], mode='concat', concat_axis=-1)

            a = Convolution2D(64 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            a = Convolution2D(96 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(a)
            b = Convolution2D(64 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            b = Convolution2D(64 // nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)
            b = Convolution2D(64 // nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)
            b = Convolution2D(96 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(b)
            x = merge([a, b], mode='concat', concat_axis=-1)

            # in original inception-resnet-v2, conv stride should be 2
            a = Convolution2D(192 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(x)
            # in original inception-resnet-v2, stride is 2
            b = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
            x = merge([a, b], mode='concat', concat_axis=-1)

            x = Activation('relu')(x)

            return x

        def inception_resnet_v2_A(x):
            shortcut = x

            a = Convolution2D(32 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            b = Convolution2D(32 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            b = Convolution2D(32 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)

            c = Convolution2D(32 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            c = Convolution2D(48 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(c)
            c = Convolution2D(64 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(c)

            x = merge([a, b, c], mode='concat', concat_axis=-1)
            x = Convolution2D(384 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='linear',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            x = merge([shortcut, x], mode='sum')
            x = Activation('relu')(x)

            return x

        def inception_resnet_v2_reduction_A(x):
            a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
            b = Convolution2D(384 // nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(x)
            c = Convolution2D(256 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            c = Convolution2D(256 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(c)
            c = Convolution2D(384 // nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(c)

            x = merge([a, b, c], mode='concat', concat_axis=-1)

            return x

        def inception_resnet_v2_B(x):
            shortcut = x

            a = Convolution2D(192 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            b = Convolution2D(128 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            b = Convolution2D(160 // nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)
            b = Convolution2D(192 // nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)

            x = merge([a, b], mode='concat', concat_axis=-1)
            x = Convolution2D(1154 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='linear',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            x = merge([shortcut, x], mode='sum')
            x = Activation('relu')(x)

            return x

        def inception_resnet_v2_reduction_B(x):
            a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
            b = Convolution2D(256 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            b = Convolution2D(288 // nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(b)
            c = Convolution2D(256 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            c = Convolution2D(288 // nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(c)
            d = Convolution2D(256 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            d = Convolution2D(288 // nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(d)
            d = Convolution2D(320 // nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                              init='he_normal', border_mode='valid', dim_ordering='tf')(d)

            x = merge([a, b, c, d], mode='concat', concat_axis=-1)

            return x

        def inception_resnet_v2_C(x):
            shortcut = x

            a = Convolution2D(192 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            b = Convolution2D(192 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)
            b = Convolution2D(224 // nb_filters_reduction_factor, 1, 3, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)
            b = Convolution2D(256 // nb_filters_reduction_factor, 3, 1, subsample=(1, 1), activation='relu',
                              init='he_normal', border_mode='same', dim_ordering='tf')(b)

            x = merge([a, b], mode='concat', concat_axis=-1)
            x = Convolution2D(2048 // nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='linear',
                              init='he_normal', border_mode='same', dim_ordering='tf')(x)

            x = merge([shortcut, x], mode='sum')
            x = Activation('relu')(x)

            return x

        # In[9]:

        img_rows, img_cols = 32, 32
        img_channels = 3

        # in original inception-resnet-v2, these are 5, 10, 5, respectively
        num_A_blocks = 1
        num_B_blocks = 1
        num_C_blocks = 1

        inputs = Input(shape=(img_rows, img_cols, img_channels))

        x = inception_resnet_v2_stem(inputs)
        for i in range(num_A_blocks):
            x = inception_resnet_v2_A(x)
        x = inception_resnet_v2_reduction_A(x)
        for i in range(num_B_blocks):
            x = inception_resnet_v2_B(x)
        x = inception_resnet_v2_reduction_B(x)
        for i in range(num_C_blocks):
            x = inception_resnet_v2_C(x)

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        with open('true-inception-report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        run = False

        if run:
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

    def test_entire_inception_with_function_bigger_aug(self):

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

        if K.image_dim_ordering() == 'th':
            print("--th")
        else:
            print("--tf")

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        x_train = np.transpose(x_train.astype('float32') / 255., (0, 2, 1, 3))
        x_test = np.transpose(x_test.astype('float32') / 255., (0, 2, 1, 3))
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

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d323311v := Conv2D @fant(32) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d323311s := Conv2D @fant(32) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d321111v := Conv2D @fant(32) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d321111s := Conv2D @fant(32) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d483311v := Conv2D @fant(48) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d483311s := Conv2D @fant(48) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d643311v := Conv2D @fant(64) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d643311s := Conv2D @fant(64) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641111v := Conv2D @fant(64) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641111s := Conv2D @fant(64) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641711v := Conv2D @fant(64) (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641711s := Conv2D @fant(64) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d647111v := Conv2D @fant(64) (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d647111s := Conv2D @fant(64) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d961111v := Conv2D @fant(96) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d961111s := Conv2D @fant(96) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d963311v := Conv2D @fant(96) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d963311s := Conv2D @fant(96) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1281111s := Conv2D @fant(128) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1601711s := Conv2D @fant(160) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1607111s := Conv2D @fant(160) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1923311v := Conv2D @fant(192) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d1923311s := Conv2D @fant(192) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1921111s := Conv2D @fant(192) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1927111s := Conv2D @fant(192) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2241311s := Conv2D @fant(224) (1, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2561111s := Conv2D @fant(256) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563311s := Conv2D @fant(256) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563111s := Conv2D @fant(256) (3, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2883322v := Conv2D @fant(288) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d2883311s:= Conv2D @fant(288) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d3203322v := Conv2D @fant(320) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d3843311v := Conv2D @fant(384) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3843311s := Conv2D @fant(384) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d3841111v := Conv2D @fant(384) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3841111slin := Conv2D @fant(384) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d3843322v := Conv2D @fant(384) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d11541111slin := Conv2D @fant(1154) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d20481111slin := Conv2D @fant(2048) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign @inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        stem2 :
            | m2d3311v
            | c2d963311v
            | Concatenate
            | c2d641111s + c2d963311v
            | c2d641111s + c2d647111s + c2d641711s + c2d963311v
            | Concatenate
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | relu

        x : stem2 x

        # layer A

        shortcut : assign x

        incA :
            | c2d321111s
            | c2d321111s + c2d323311s
            | c2d321111s + c2d483311s + c2d643311s
            | Concatenate
            | assign shortcut
            | c2d3841111slin
            | sum
            | relu

        x : incA x

        incA_red :
            | m2d3322v
            | c2d3843322v
            | c2d2561111s + c2d2563311s + c2d3843322v
            | Concatenate

        x : incA_red x

        #layer B

        shortcut : assign x

        incB : 
            | c2d1921111s
            | c2d1281111s + c2d1601711s + c2d1927111s
            | Concatenate
            | assign shortcut
            | c2d11541111slin
            | sum
            | relu

        x : incB x

        incB_red :
            | m2d3322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883322v
            | c2d2561111s + c2d2883311s + c2d3203322v
            | Concatenate

        x : incB_red x

        shortcut : assign x

        incC : 
            | c2d1921111s
            | c2d1921111s + c2d2241311s + c2d2563111s
            | Concatenate
            | assign shortcut
            | c2d20481111slin
            | sum
            | relu

        x : incC x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

        x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        predictions = Dense(nb_classes, activation='softmax')(x)

        model = Model(input=inputs, output=predictions)

        # In[10]:

        model.summary()

        # with open('my-inception-report.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

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

        run = False

        if run:
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