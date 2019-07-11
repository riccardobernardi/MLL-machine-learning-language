from lark import Tree, Token
from termcolor import cprint

from mll.dispatcher import Dispatcher
from mll.mlltranspiler import MLL
from mll.utils import clean_tok, apply, match, isTree, istok, escape, filter, map


class SequentialModel:
    def __init__(self, mll: MLL):
        self.mll = mll

    ###################################################################
    #                      SEQUENTIAL MODEL                           #
    ###################################################################

    def traduce_sequential(self, t: Tree):
        # t.children[2:] = escape(t.children[2:])
        t.children[2:] = self.mll.put_macros(t.children[2:])

        apply(t, lambda x: x, self.mll.select_imported_libraries)

        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)

        self.mll.models[clean_tok(t.children[0]).value] = 0
        self.mll.select_imported_libraries(Token("ID", "Sequential"))

        self.mll.ordered_models.append(clean_tok(t.children[0]).value)

        # cprint(t.children[2:],"green")

        branches = map(Dispatcher(self.mll,"sequential").transform,t.children[2:])

        # cprint("branches:"+str(branches),"blue")

        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "="), Token("ID", "Sequential"), Token("LP", "("),
                     Token("LSP", "[")]
                    +
                    branches
                    +
                    [Token("RSP", "]"), Token("RP", ")"), Token("WS", "\n\n")])