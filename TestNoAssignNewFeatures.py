import unittest

from mll.mlltranspiler import MLL


class TestNoAssignNewFeatures(unittest.TestCase):

    def test_simple_model_function_wo_params_wo_AT(self):
        skmodel4 = """
        model : Flatten
        """

        self.mll = MLL(skmodel4)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print(type(sclf))

    def test_simple_model_function_wo_params_w_AT(self):
        skmodel4 = """
        model : @Flatten
        """

        self.mll = MLL(skmodel4)
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print(type(sclf))

    def test_seq_model_function_wo_params_wo_AT(self):
        def fant(x):
            return x

        skmodel4 = """

        init $ he_normal
        border_mode $ valid or same
        dim_ordering $ tf
        activation $ relu or linear

        val_rel := he_normal valid tf relu
        m2dv := valid tf

        c2d1923311v := Conv2D ( fant ) (3, 3) val_rel with subsample=(1,1) 

        model : c2d1923311v
        """

        self.mll = MLL(skmodel4, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print(type(sclf))

    def test_seq_model_function_wo_params_w_AT(self):
        def fant():
            return 32

        skmodel4 = """

        init $ he_normal
        border_mode $ valid or same
        dim_ordering $ tf
        activation $ relu or linear

        val_rel := he_normal valid tf relu
        m2dv := valid tf

        c2d1923311v := Conv2D ( @fant ) (3, 3) val_rel with subsample=(1,1) 

        model : c2d1923311v
        """

        self.mll = MLL(skmodel4, locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print(type(sclf))

    def test_sk_simple_2_wo_AT(self):

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

        print(type(sclf))


    def test_sk_simple_2_w_AT(self):

        def fant(x):
            return x

        skmodel4 = """criterion $ gini or entropy

        rf_clf  : RandomForestClassifier 10 entropy
        knn_clf : KNeighborsClassifier 2
        svc_clf : SVC with C=10000.0
        rg_clf  : RidgeClassifier ( fant 0.1 )
        dt_clf  : DecisionTreeClassifier gini
        lr      : @LogisticRegression
        sclf : classifier StackingClassifier with classifiers = [ rf_clf, dt_clf, knn_clf, svc_clf, rg_clf ] meta_classifier = lr

        """

        self.mll = MLL(skmodel4,locals())
        self.mll.start()
        print(self.mll.get_string())
        self.mll.execute()
        sclf = self.mll.last_model()

        print(type(sclf))

    def test_func_app(self):
        ext = 384

        def f():
            return (1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D 32 (1,1) subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : x

        incA1 :
            | c2d38411s_ext
            | shortcut
            | Concatenate

        x : incA1 x

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()

    def test_func_app_summa(self):
        ext = 384

        def f():
            return (1,1)

        #print(locals())

        inc = """
        conv2d := Conv2D

        c2d38411s_ext := Conv2D 32 (1,1) subsample=(1,1)

        x : Input with shape = (32,32,3)

        shortcut : x

        incA1 :
            | c2d38411s_ext
            | shortcut
            | Concatenate

        x +: incA1

        """

        self.mll = MLL(inc, locals())
        self.mll.start()
        print(self.mll.get_string())
        #print(self.mll.import_from_glob)
        self.mll.execute()










if __name__ == '__main__':
    unittest.main()
