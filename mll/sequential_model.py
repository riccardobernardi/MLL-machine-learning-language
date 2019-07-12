import copy

from lark import Tree, Token
from termcolor import cprint

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

        t.children[2:] = self.mll.put_macros(t.children[2:])

        apply(t, lambda x: x, self.mll.select_imported_libraries)

        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)

        self.mll.models[clean_tok(t.children[0]).value] = t
        self.mll.select_imported_libraries(Token("ID", "Sequential"))

        self.mll.ordered_models.append(clean_tok(t.children[0]).value)

        branches = map(Dispatcher(self.mll,"sequential").transform,t.children[2:])

        # print(branches)
        # print(descend_left(Tree("e",branches)))
        # print(self.mll.env.keys())
        # print("Conv2D" in self.mll.env.keys())

        # cprint("ATTENTION SEQ", "red")

        #va in self loop, fixare
        if self.mll.isInner==False:
            #se il tipo può essere direttamente inferito:
            a = 1
            try:
                # se il modello è complesso allora darà errore che non esiste perchè non è ancora stato elaborato
                a = MLL("model:"+clean_tok(descend_left(branches[0])).value.replace("models['","").replace("']","")+"\n\n", self.mll.env).inner().execute().last_model()
            except:
                # a = MLL("model:"+clean_tok(descend_left(branches[0])).value.replace("models['","").replace("']","")+"\n\n", MLL()).inner().execute().last_model()
                # print(descend_left(t.children[2]))
                b = self.mll.function_trees[clean_tok(descend_left(t.children[2])).value.replace("models['","").replace("']","")].children[2:][0]
                b = b.children[0]
                # print(b)
                a = MLL("model:"+clean_tok(b).value.replace("models['","").replace("']","")+"\n\n", self.mll.env).inner().execute().last_model()

            # print(a)

            if ("classifier" in str(a) or "regressor" in str(a)) and "sklearn" in str(a):
                # è un modello di stacking
                pass

                return Tree(t.data,[
                    Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                    Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "="), Token("ID", "StackingClassifier"), Token("LP", "("),
                    Token("ID","classifiers"), Token("EQ","="), Token("LSP","["),
                    Token("LSP", "[")]
                            +
                            branches
                            +
                            [Token("RSP", "]"), Token("RP", ")"), Token("WS", "\n\n")
                ])

            if "classifier" not in str(a).lower() and "regressor" not in str(a).lower() and "sklearn" in str(a).lower():
                # è un modello di pipeline
                self.mll.select_imported_libraries(Token("ID","Pipeline"))
                return Tree(t.data,
                            [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                             Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "="), Token("ID", "Pipeline"),
                             Token("LP", "("),
                             Token("LSP", "[")]
                            # devo trasformare da [a,b,c] -> [("a",a),("b",b),("c",c)]
                            +
                            branches
                            +
                            [Token("RSP", "]"), Token("RP", ")"), Token("WS", "\n\n")])

        # in alternativa è un modello sequenziale
        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "="), Token("ID", "Sequential"), Token("LP", "("),
                     Token("LSP", "[")]
                    +
                    branches
                    +
                    [Token("RSP", "]"), Token("RP", ")"), Token("WS", "\n\n")])