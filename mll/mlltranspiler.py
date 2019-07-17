from lark import Lark, Token
from lark.tree import Tree
from termcolor import cprint

from mll.grammar_revision import get_rev_grammar
from mll.superMLL import superMLL
from mll.utils import scrivi, map, match, flatten, filter, group, clean_tok, \
    alphabet, \
    OR, visit, istok, apply, isTree, Toggler, list_types_list, escape, create_macro_mod, create_macro_exp, \
    create_macro_pip, leaves_before, leaves_after, presentation

import os
import sys

import warnings

warnings.filterwarnings("ignore")


class MLL(superMLL):

    ###################################################################
    #                           START                                 #
    ###################################################################

    def __init__(self, program: str, env={}) -> None:
        superMLL.__init__(self,program, env)

    def start(self):
        # presentazione del progetto con un frontespizio
        presentation()

        self.create_available_imports()
        self.string = self.transpile(self.program)

        return self

    def inner(self):
        self.isInner = True

        # old_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')

        self.create_available_imports()
        self.string = self.transpile(self.program)

        return self

    def transpile(self, program: str) -> str:

        if self.isInner == False:

            # print("                              DEBUG")
            # print("###############################################################")
            pass

        parser = Lark(get_rev_grammar(), start='mll')

        self.before_tree = parser.parse(program)
        # print(tree_depth(self.before_tree))
        # print(leaves_before(self.before_tree))

        self.after_tree = Tree(self.before_tree.data, self.transform(self.before_tree.children))
        # print(tree_depth(self.before_tree))
        # print(self.after_tree)
        # print(leaves_after(self.after_tree))

        s = scrivi(self.used_libraries) + scrivi(self.after_tree)

        if self.isInner == False:

            print("###############################################################")
            print("                           POSTCONDIZIONI")
            # print("le post-condizioni sono relative solo a alla transpilazione e non all' esecuzione")
            # print("occhio alla +: potrebbe confondersi con una e -> e + e")
            cprint("macros: "+str(self.macros.keys()),"blue")
            cprint("parmacs: " + str(self.parmacs.keys()), "blue")
            cprint("models: " + str(self.models.keys()), "blue")
            cprint("MLL : Python = 1 : "+str(leaves_after(self.after_tree)/leaves_before(self.before_tree)),"yellow")
            print("###############################################################")

            print("                             PROGRAM")
            print("###############################################################")
        return s

    ###################################################################
    #                        MAIN DISPATCHER                          #
    ###################################################################

    def translate_tree(self, t: Tree) -> Tree:

        # print(t)

        if t.data == "mll":
            return Tree(t.data, self.transform(t.children))

        if t.data == "pyt":
            m = apply(t,lambda x:x, clean_tok)
            m.children = m.children + [Token("WS","\n")]
            return m

        if t.data == "model":
            return self.translate_model(t)

        if t.data == "comp":
            return Tree(t.data, escape(t.children))

        if t.data == "comment":
            return None

        if t.data == "parmac":
            # cprint("entro in parmac","yellow")
            self.insert_parmac(t)

        if t.data == "summa":
            # cprint("entro in parmac","yellow")
            from mll.simple_model import SimpleModel
            from mll.dispatcher import Dispatcher
            rest = Dispatcher(self).translate_e(Tree("e",[
                Token("ID", clean_tok(t.children[2]).value ),
                Tree("e", [Token("ID",clean_tok(t.children[0]).value)])
                ]))
            print(rest)
            m = apply([Token("ID", clean_tok(t.children[0]).value), Token("EQ", "=")] + rest.children, lambda x: x,
                  self.substitute_model)
            print(m)
            m = m + [Token("WS","\n\n")]
            return m

        # if t.data == "macro_mod":
        #     create_macro_mod(self,t)
        #
        # if t.data == "macro_exp":
        #     create_macro_exp(self,t)
        #
        # if t.data == "macro_pip":
        #     create_macro_pip(self,t)

        if t.data == "macro":
            create_macro_pip(self,t)

    def translate_list(self, t: list):
        return filter(lambda x: x is not None, [self.transform(x) for x in t])

    def transform(self, t: object) -> object:
        if isinstance(t, Tree):
            return self.translate_tree(t)
        if isinstance(t, type([])):
            return self.translate_list(t)

        raise Exception("Non esiste questo caso nella fun transform: ", type(t))

    ###################################################################
    #                       RULES DISPATCHER                          #
    ###################################################################

    def translate_model(self, t: Tree):

        if istok(t.children[2]) and t.children[2].type == "RR":
            self.regressors.append(t.children[1].value)
            t.children.pop(2)
        if istok(t.children[2]) and t.children[2].type == "CC":
            self.classifiers.append(t.children[1].value)
            t.children.pop(2)

        branches = list(group(t.children, "PI"))

        if len(branches) == 1:
            plus = lambda x: True if x.type == "PLUS" else False
            if visit(t, plus, OR):
                # cprint("SEQUENTIAL","red")
                from mll.sequential_model import SequentialModel
                return SequentialModel(self).traduce_sequential(t)
            else:
                # cprint("SIMPLE", "red")
                from mll.simple_model import SimpleModel
                return SimpleModel(self).traduce_simple_model(t)
        else:
            # cprint("FORKED", "red")
            from mll.forked_model import ForkedModel
            return ForkedModel(self).traduce_forks(t)


