from lark import Tree, Token
import inspect

from mll.dispatcher import Dispatcher
from mll.mlltranspiler import MLL
from mll.utils import clean_tok, apply, isTree, map, escape, istok, match, filter, contained_in_imported_libraries_mod


# def get_params_for_function(func_name, mll):
#     print("get_params_for_function")
#     arr = []
#     # step 1: chiedo i parametri all utente e li metto in una lista, i nomi dei parametri si trovano nella eval di func_name
#
#     a = MLL("model:" + str(func_name) + "\n\n",
#             mll.env).inner().execute().last_model()
#     b = inspect.getfullargspec(a)
#     print(b)
#
#     for i in b:
#         c = input(i)
#         if (len(c) > 0) and (c != "\n"):
#             arr.append(i+"="+c)
#
#     return arr
#
#
# def add_params(l, g, mll):
#     print("add_params")
#     # step 1: per ogni l in cui g è true allora modifico g[i] aggiungendoci i parametri
#     print(len(l))
#     for i in range(len(l)):
#         if g[i] == True:
#             l[i] = l[i] + "(" + ",".join(get_params_for_function(l[i], mll)) + ")"
#     return l

class SimpleModel:

    def __init__(self, mll: MLL):
        self.mll = mll

    ###################################################################
    #                        SIMPLE MODEL                             #
    ###################################################################

    def traduce_simple_model(self, t: Tree):

        print("confirm simple")

        t.children[2:] = self.mll.put_macros(t.children[2:])
        t.children[2:] = apply(t.children[2:], lambda x: x, clean_tok)
        t.children[2:] = Dispatcher(self.mll,"simple").transform(t.children[2:])

        print(t.children[2:])

        apply(t.children[2:], lambda x: x, self.mll.select_imported_libraries)
        # g = apply(t.children[2:], lambda x: x, self.mll.contained_in_imported_libraries)
        # print("names that can be imported", g)
        # l = add_params(t.children[2:], g, self.mll)
        # print("modified list", l)
        #print(locals())
        if not self.mll.isInner:
            t.children[2:] = apply(t.children[2:], lambda x: x, contained_in_imported_libraries_mod, self.mll)
            print("mods to initial vector: "+str(t.children[2:]))

        self.mll.models[clean_tok(t.children[0]).value] = t
        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)

        self.mll.ordered_models.append(clean_tok(t.children[0]).value)

        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "=")]
                    +
                    t.children[2:]
                    + [Token("WS", "\n\n")])