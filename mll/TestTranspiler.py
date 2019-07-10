import unittest

from keras import Input
from lark import Token, Tree
from termcolor import cprint

from mll.forked_model import ForkedModel
from mll.mlltranspiler import MLL
from mll.simple_model import SimpleModel
from mll.utils import split, group, map, tree_depth, MAX, reduce, match, scrivi, visit, OR, apply, stampa


class TestTranspiler(unittest.TestCase):

    def test_mll(self):
        inc = """conv2d := Conv2D

        stem2 : m2d3311
            | c2d9630
            | c2d9631
            | c2d9632
            | c2d9633
            | Concatenate
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_dag_plus(self):
        inc = """relu := Activation relu
        m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu

        stem2 : m2d3311
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | Concatenate
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_split(self):
        a = [Token("ID","ciao"),Token("PI","|"),Token("ID","ciao"),Token("PI","|"),Token("ID","ciao"),Token("PI","|"),Token("ID","ciao"),Token("PI","|"),Token("ID","ciao")]

        print(split(a,"PI"))

    def test_function(self):
        inc = """model : RandomForestClassifier 10
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_group(self):
        a = [Token("ID", "ciao"), Token("PI", "|"), Token("ID", "ciao"), Token("PI", "|"), Token("ID", "ciao"),
             Token("PI", "|"), Token("ID", "ciao"), Token("PI", "|"), Token("ID", "ciao")]

        print(list(group(a, 'PI')))

    def test_traduce_forks(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS="PLUS"
        e="e"
        PI="PI"
        a = Tree(model, [Token(ID, 'stem2 '), Token(COLON, ': '), Tree(e, [Token(ID, 'm2d3311\n            ')]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9631 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9632 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9633 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Token(ID, 'Concatenate\n            ')])])
        mll = MLL("")
        print(mll.translate_model(a))

    def test_traduce_token_cleansing(self):
        mll = MLL("")
        print(SimpleModel(mll).translate_token_simple(Token("ID", "\t\t\t ciao")))

    def test_map(self):
        a = [Token("ID", "ciao"), Token("ID", "ciao"), Token("ID", "ciao"), Token("ID", "ciao"), Token("ID", "ciao")]

        print(map(lambda x: x.value if type(x) == Token else x, map(lambda x: x.value, a, "CO", ",")))

    def test_tree_depth_complex(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        a = Tree(model, [Token(ID, 'stem2 '), Token(COLON, ': '), Tree(e, [Token(ID, 'm2d3311\n            ')]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9631 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9632 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9633 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Token(ID, 'Concatenate\n            ')])])
        print(tree_depth(a))

    def test_tree_depth_one(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        a = Tree(model, [Token(ID, 'stem2 ')])
        print(tree_depth(a))

    def test_tree_depth_zero(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        a = Tree(model, [Token(ID, 'stem2 ')])
        print(tree_depth(a))

    def test_tree_depth_due(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        a = Tree(model, [Token(ID, 'stem2 '), Tree("e", [Token("ID","nome")])])
        print(tree_depth(a))

    def test_max(self):
        print(MAX(1,2))
        print(MAX(2, 2))
        print(MAX(1, 0))

    def test_reduction_on_list(self):
        print(reduce(MAX, [1,2,3,4,5,6]))

    def test_match(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        t = Tree(e, [Tree(e, [Token(ID, 'm2d3311 ')]), Token(PLUS, '+ '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])])])
        print(match(t.children, [1], ["PLUS"]))
        print(len(t.children) == 3)

    def test_create_imports(self):
        mll = MLL("")
        mll.create_available_imports()
        print(mll.available_libraries)

    def test_existence_of_certain_imports(self):
        mll = MLL("",{})
        mll.create_available_imports()
        print(mll.available_libraries)
        print("RandomForestClassifier" in mll.available_libraries)
        print("Sequential" in mll.available_libraries)

    def test_substring(self):
        mll = MLL("")
        mll.create_available_imports()
        print("a" in "ciao")
        print("Conv2D" in mll.available_libraries)
        for i in mll.available_libraries:
            if "a" in i:
                cprint(i, "blue")

        # from sklearn.ensemble import RandomForestClassifier

        for i in mll.available_libraries:
            if "rand" in i:
                cprint(i, "blue")

    def test_sequential_model_with_macros(self):
        inc = """
            relu := Activation relu
            
            m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
            c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
            c2d9633init := Conv2D 96 (3, 3) input_shape=(32,32,3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
            
            stem2 : c2d9633init m2d3311 + c2d9633 + c2d9633
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_new_match(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        t = Tree(e, [Tree(e, [Token(ID, 'm2d3311 ')]), Token(PLUS, '+ '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]),
                                                                                   Token(PLUS, '+ '), Tree(e, [
                Token(ID, 'c2d9630\n            ')])])])
        print(match(t.children, [], ["PLUS"]))
        print(len(t.children))

        t = Tree(e, [Tree(e, [Token(ID, 'm2d3311 ')]), Tree(e, [Token(ID, 'm2d3311 ')]), Token(PLUS, '+ '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]),
                                                                                   Token(PLUS, '+ '), Tree(e, [
                Token(ID, 'c2d9630\n            ')])])])
        print(match(t.children, [], ["PLUS"]))
        print(len(t.children))

    def test_array_bound(self):
        a = [1,2,3,4]
        print(a[1:len(a) - 1])
        print(a[len(a)-1])

    def test_escape(self):

        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"

        t = Tree(e, [Tree(e, [Token(ID, 'm2d3311 ')]), [Token(ID, 'relu ')], Tree(e, [Token(ID, 'm2d3311 ')]), Token(PLUS, '+ '),
                     Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]),
                              Token(PLUS, '+ '), Tree(e, [
                             Token(ID, 'c2d9630\n            ')])])])

    def test_forked_model_with_macros(self):
        inc = """relu := Activation relu
                    m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
                    c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu
                    stem2 : m2d3311 + c2d9633 + c2d9633
                        | m2d3311 + c2d9633 + c2d9633
                        | m2d3311 + c2d9633 + c2d9633
        """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_traduce_forks_format_branch(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"

        mll = MLL("")
        mll.current_branch = 1

        tree = Tree(model, [Token(ID, 'stem2 '), Token(COLON, ': '), Tree(e, [Token(ID, 'm2d3311\n            ')]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9631 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9632 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9633 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Token(ID, 'Concatenate\n            ')])])
        branches = [[Tree(e, [Token(ID, 'm2d3311\n            ')])], [Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]),
                                                                                 Token(PLUS, '+ '), Tree(e, [
                Token(ID, 'c2d9630\n            ')])])], [Tree(e, [Tree(e, [Token(ID, 'c2d9631 ')]), Token(PLUS, '+ '),
                                                                   Tree(e, [Token(ID, 'c2d9630\n            ')])])], [
                          Tree(e, [Tree(e, [Token(ID, 'c2d9632 ')]), Token(PLUS, '+ '),
                                   Tree(e, [Token(ID, 'c2d9630\n            ')])])], [Tree(e, [
            Tree(e, [Token(ID, 'c2d9633 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])])],
                      [Tree(e, [Token(ID, 'Concatenate\n            ')])]]

        a = ForkedModel(mll).traduce_forks(tree)

        print(a)
        print(scrivi(a))

    def ff(self,el:int,s:str):
        print(el,":",s)

    def test_opt_map(self):
        map(self.ff,[1,2,3],"","","ciaooooo")

    def test_dag_without_macros(self):
        inc = """stem2 : MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf' + MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
                    | MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf' + MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
                    | MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf' + MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
                """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_sk_1(self):
        skmodel4 = """criterion $ gini or entropy
        
        rf_clf  : RandomForestClassifier 10 entropy
        knn_clf : KNeighborsClassifier 2
        svc_clf : SVC with C=10000.0
        rg_clf  : RidgeClassifier 0.1
        dt_clf  : DecisionTreeClassifier gini
        lr      : LogisticRegression
        sclf : classifier StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        """

        self.mll = MLL(skmodel4)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        # train, test = get_data()

        # sclf.fit(train, test)

        # scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        # print(scores.mean(), scores.std())

    def test_visit(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        tree = Tree(model, [Token(ID, 'stem2 '), Token(COLON, ': '), Tree(e, [Token(ID, 'm2d3311\n            ')]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9630 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9631 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9632 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Tree(e, [Token(ID, 'c2d9633 ')]), Token(PLUS, '+ '), Tree(e, [Token(ID, 'c2d9630\n            ')])]), Token(PI, '| '), Tree(e, [Token(ID, 'Concatenate\n            ')])])

        print(visit(tree,lambda x: x,lambda x,y: y ))

    def test_visit_2(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        tree = Tree(model, [Token("ID","ciao"),Token("ID","nope")])

        print(visit(tree,lambda x: x,lambda x,y: y))

    def test_visit_3(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        tree = Tree(model, [Token("ID","ciao"),Token("PLUS","+"),Token("ID","nope")])

        plus = lambda x: True if x.type == "PLUS" else False
        print(visit(tree,plus,OR))

    def test_apply(self):
        model = "model"
        ID = "ID"
        COLON = "COLON"
        PLUS = "PLUS"
        e = "e"
        PI = "PI"
        tree = Tree(model, [Token("ID","ciao"),Token("PLUS","+"),Token("ID","nope")])

        print(apply(tree, lambda x:x, lambda x : Token("ID",",") if x.type=="PLUS" else x))

    def test_dag_right_concat(self):
        inc = """relu := Activation relu
        m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu

        stem2 : m2d3311
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | Concatenate
            | m2d3311
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | Concatenate
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_very_simple_net(self):
        inc = """relu := Activation relu
        m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu

        stem2 : c2d9633 + c2d9633
            | c2d9633 + c2d9633
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

    def test_net_with_bindings(self):
        inc = """relu := Activation relu
        m2d3311 := MaxPooling2D (3, 3) strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        c2d9633 := Conv2D 96 (3, 3) subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' + relu

        stem2 : right ->
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | left ->
            | c2d9633 + c2d9633
            | c2d9633 + c2d9633
            | concatenate -> right left
            """

        self.mll = MLL(inc)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()

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

        def assign(x):
            return x

        m = 4

        inc = """
        conv2d := Conv2D
        seq := Sequential
        drop := Dropout
        dense := Dense
        flatten := Flatten
        soft := Activation 'softmax'

        c2d323311v := Conv2D ( fant 32 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'

        c2d643311s := Conv2D ( fant 64 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'

        # Input layer

        x : assign inputs

        # Layer stem di entrata dell input

        stem1 :
            | c2d323311v + c2d323311v + c2d643311s

        x : stem1 x

        """

        print(locals().keys())

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()
        print(type(x))

    def test_LP_e_RP_true(self):
        t = Tree("e",[Token("LP","("),Tree("e",[Token("ID","ciao")]),Token("RP",")")])
        stampa(t)
        print()
        print("match:",match(t.children, [0, len(t.children)-1], ["LP", "RP"]))

    def test_LP_e_RP_false(self):
        t = Tree("e",[Token("NUMBER","10")])
        stampa(t)
        print()
        print("match:",match(t.children, [0, len(t.children)-1], ["LP", "RP"]))

    def test_env_inputs(self):

        # we reduce # filters by factor of 8 compared to original inception-v4
        nb_filters_reduction_factor = 8

        img_rows, img_cols = 32, 32
        img_channels = 3

        inputs = Input(shape=(img_rows, img_cols, img_channels))

        def fant(x):
            return x // nb_filters_reduction_factor

        m = 4

        self.mll = MLL("model: assign inputs", locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        print("qui printo il tipo per controllare che esista")
        exec("print(type(inputs))",self.mll.env)
        x = self.mll.last_model()

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
        soft := Activation softmax
        relu := Activation relu

        c2d323311v := Conv2D ( fant 32 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d323311s := Conv2D ( fant 32 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d321111v := Conv2D ( fant 32 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d321111s := Conv2D ( fant 32 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d483311v := Conv2D ( fant 48 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d483311s := Conv2D ( fant 48 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d643311v := Conv2D ( fant 64 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d643311s := Conv2D ( fant 64 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641111v := Conv2D ( fant 64 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641111s := Conv2D ( fant 64 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641711v := Conv2D ( fant 64 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641711s := Conv2D ( fant 64 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d647111v := Conv2D ( fant 64 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d647111s := Conv2D ( fant 64 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d961111v := Conv2D ( fant 96 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d961111s := Conv2D ( fant 96 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d963311v := Conv2D ( fant 96 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d963311s := Conv2D ( fant 96 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1281111s := Conv2D ( fant 128 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1601711s := Conv2D ( fant 160 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1607111s := Conv2D ( fant 160 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1923311v := Conv2D ( fant 192 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d1923311s := Conv2D ( fant 192 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1921111s := Conv2D ( fant 192 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1927111s := Conv2D ( fant 192 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2241311s := Conv2D ( fant 224 ) (1, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2561111s := Conv2D ( fant 256 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563311s := Conv2D ( fant 256 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563111s := Conv2D ( fant 256 ) (3, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2883322v := Conv2D ( fant 288 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d2883311s:= Conv2D ( fant 288 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d3203322v := Conv2D ( fant 320 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d3843311v := Conv2D ( fant 384 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3843311s := Conv2D ( fant 384 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d3841111v := Conv2D ( fant 384 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3841111slin := Conv2D ( fant 384 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d3843322v := Conv2D ( fant 384 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d11541111slin := Conv2D ( fant 1154 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d20481111slin := Conv2D ( fant 2048 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign inputs

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

        """

        print("inputs in locals?:", 'inputs' in locals())
        print(locals().keys())

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

    def test_entire_inception_with_function_bigger_simpler(self):

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
        relu := Activation relu

        c2d323311v := Conv2D ( fant 32 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d323311s := Conv2D ( fant 32 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d321111v := Conv2D ( fant 32 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d321111s := Conv2D ( fant 32 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d483311v := Conv2D ( fant 48 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d483311s := Conv2D ( fant 48 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d643311v := Conv2D ( fant 64 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d643311s := Conv2D ( fant 64 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641111v := Conv2D ( fant 64 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641111s := Conv2D ( fant 64 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d641711v := Conv2D ( fant 64 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d641711s := Conv2D ( fant 64 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d647111v := Conv2D ( fant 64 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d647111s := Conv2D ( fant 64 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d961111v := Conv2D ( fant 96 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d961111s := Conv2D ( fant 96 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d963311v := Conv2D ( fant 96 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d963311s := Conv2D ( fant 96 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1281111s := Conv2D ( fant 128 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1601711s := Conv2D ( fant 160 ) (1, 7) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1607111s := Conv2D ( fant 160 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d1923311v := Conv2D ( fant 192 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d1923311s := Conv2D ( fant 192 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1921111s := Conv2D ( fant 192 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d1927111s := Conv2D ( fant 192 ) (7, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2241311s := Conv2D ( fant 224 ) (1, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2561111s := Conv2D ( fant 256 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563311s := Conv2D ( fant 256 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d2563111s := Conv2D ( fant 256 ) (3, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d2883322v := Conv2D ( fant 288 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d2883311s:= Conv2D ( fant 288 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'


        c2d3203322v := Conv2D ( fant 320 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d3843311v := Conv2D ( fant 384 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3843311s := Conv2D ( fant 384 ) (3, 3) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='relu'
        c2d3841111v := Conv2D ( fant 384 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'
        c2d3841111slin := Conv2D ( fant 384 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d3843322v := Conv2D ( fant 384 ) (3, 3) with subsample=(2,2) init='he_normal' border_mode='valid' dim_ordering='tf' activation='relu'


        c2d11541111slin := Conv2D ( fant 1154 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        c2d20481111slin := Conv2D ( fant 2048 ) (1, 1) with subsample=(1,1) init='he_normal' border_mode='same' dim_ordering='tf' activation='linear'


        m2d3311v := MaxPooling2D (3, 3) with strides=(1, 1) border_mode='valid' dim_ordering ='tf'
        m2d3322s := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='same' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'
        m2d3322v := MaxPooling2D (3, 3) with strides=(2, 2) border_mode='valid' dim_ordering ='tf'


        # Input layer

        x : assign inputs

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
            | Add
            | relu

        x : incA x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        exec("print(type(inputs))",self.mll.env)
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

    def test_func_app(self):
        ext = 384

        def f():
            return (1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D 32 (1,1) subsample=(1,1)

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
        init $ he_normal
        border_mode $ valid or same
        dim_ordering $ tf
        activation $ relu or linear

        val_rel := he_normal valid tf relu
        m2dv := valid tf

        c2d1923311v := Conv2D ( fant 192 ) (3, 3) val_rel with subsample=(1,1) 

        m2d3311v := MaxPooling2D (3, 3) m2dv with strides=(1, 1)

        # Input layer

        x : assign inputs

        # Layer stem di entrata dell input

        stem2 :
            | c2d1923311v
            | m2d3311v
            | Concatenate
            | Activation 'relu'

        x : stem2 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        x = self.mll.last_model()

    def test_sk_simple_2(self):

        def fant(x):
            return x

        skmodel4 = """criterion $ gini or entropy

        rf_clf  : RandomForestClassifier 10 entropy
        knn_clf : KNeighborsClassifier 2
        svc_clf : SVC with C=10000.0
        rg_clf  : RidgeClassifier ( fant 0.1 )
        dt_clf  : DecisionTreeClassifier gini
        lr      : LogisticRegression
        sclf : classifier StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        """

        self.mll = MLL(skmodel4,locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        # train, test = get_data()

        # sclf.fit(train, test)

        # scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        # print(scores.mean(), scores.std())


    def test_simple_conv(self):

        def fant(x):
            return x

        skmodel4 = """
        
        init $ he_normal
        border_mode $ valid or same
        dim_ordering $ tf
        activation $ relu or linear

        val_rel := he_normal valid tf relu
        m2dv := valid tf

        c2d1923311v := Conv2D ( fant 192 ) (3, 3) val_rel with subsample=(1,1) 
        
        model : c2d1923311v
        """

        self.mll = MLL(skmodel4,locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        # train, test = get_data()

        # sclf.fit(train, test)

        # scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        # print(scores.mean(), scores.std())

    def simple_conv_new(self):

        def fant(x):
            return x

        skmodel4 = """init $ he_normal
        border_mode $ valid or same
        dim_ordering $ tf
        activation $ relu or linear

        val_rel e: he_normal valid tf relu
        m2dv e: valid tf

        c2d1923311v m: Conv2D ( fant 192 ) (3, 3) val_rel with subsample=(1,1) 

        model : c2d1923311v
        """

        self.mll = MLL(skmodel4, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print("[",self.mll.macros,"]")

        # train, test = get_data()

        # sclf.fit(train, test)

        # scores = model_selection.cross_val_score(sclf, train, test, cv=3, scoring='accuracy')
        # print(scores.mean(), scores.std())


if __name__ == '__main__':
    unittest.main()
