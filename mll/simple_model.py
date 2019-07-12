from lark import Tree, Token

from mll.dispatcher import Dispatcher
from mll.mlltranspiler import MLL
from mll.utils import clean_tok, apply, isTree, map, escape, istok, match, filter


class SimpleModel:

    def __init__(self, mll: MLL):
        self.mll = mll

    ###################################################################
    #                        SIMPLE MODEL                             #
    ###################################################################

    def traduce_simple_model(self, t: Tree):

        t.children[2:] = self.mll.put_macros(t.children[2:])
        t.children[2:] = apply(t.children[2:], lambda x: x, clean_tok)
        t.children[2:] = Dispatcher(self.mll,"simple").transform(t.children[2:])

        apply(t, lambda x: x, self.mll.select_imported_libraries)

        self.mll.models[clean_tok(t.children[0]).value] = t
        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)

        self.mll.ordered_models.append(clean_tok(t.children[0]).value)

        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "=")]
                    +
                    t.children[2:]
                    + [Token("WS", "\n\n")])