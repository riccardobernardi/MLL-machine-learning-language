from lark import Tree, Token

from mll.dispatcher import Dispatcher
from mll.mlltranspiler import MLL
from mll.utils import clean_tok, apply, map, descend_left, scrivi


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

        branches = map(Dispatcher(self.mll,"sequential").transform,t.children[2:])

        print(branches)
        print(descend_left(Tree("e",branches)))
        # print(self.mll.env.keys())
        # print("Conv2D" in self.mll.env.keys())

        #va in self loop, fixare
        if self.inner==False:
            print(MLL("model:"+clean_tok(descend_left(branches[0])).value.replace("models['","").replace("']","")+"\n\n",MLL(self.mll.program).start().execute().models).start().execute().last_model())

        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "="), Token("ID", "Sequential"), Token("LP", "("),
                     Token("LSP", "[")]
                    +
                    branches
                    +
                    [Token("RSP", "]"), Token("RP", ")"), Token("WS", "\n\n")])