from lark import Tree, Token

from mll.mlltranspiler import MLL
from mll.utils import clean_tok, apply, isTree, map, escape, istok, match, filter


class SimpleModel:

    def __init__(self, mll: MLL):
        self.mll = mll

    ###################################################################
    #                        *SIMPLE* DISPATCHER                      #
    ###################################################################

    def transform_simple(self, t: object):
        if isinstance(t, Token):
            return self.translate_token_simple(t)
        elif isinstance(t, Tree):
            return self.translate_tree_simple(t)
        elif isinstance(t, type([])):
            return self.translate_list_simple(t)
        elif t is None:
            return None
        else:
            raise Exception("Non esiste questo caso nella fun transform: ", type(t), t)

    def translate_tree_simple(self, t: Tree) -> Tree:
        if t.data == "comp":
            return Tree(t.data, self.transform_simple(t.children))

        if t.data == "e" or t.data == "macro":
            return self.translate_e_simple(t)

        return t

    def translate_token_simple(self, t: Token) -> object:
        t = self.mll.put_macros(t)
        self.mll.select_imported_libraries(t)
        return t

    def translate_list_simple(self, t: list):
        return filter(lambda x: x is not None, [self.transform_simple(x) for x in t])

    def translate_e_simple(self, t: Tree):
        # e ::= ID
        if match(t.children, [0], ["ID"]) and len(t.children) == 1:
            return Tree(t.data, [self.transform_simple(t.children)])

        # e ::= ID e+
        if match(t.children, [0], ["ID"]) and len(t.children) > 1 and not match(t.children, [], ["PLUS"]):
            # print("questo viene dal simple")
            # cprint(t.children,"green")
            return Tree(t.data,
                        [t.children[0], Token("LP", "(")] +
                        map(self.transform_simple, t.children[1:len(t.children)], "CO", ",") +
                        [Token("RP", ")")])
        # e ::= LP e RP
        if match(t.children, [0, len(t.children) - 1], ["LP", "RP"]) and len(t.children)>=3:
            # print(t.children)
            return Tree(t.data, [self.transform_simple(t.children[1])])
        if match(t.children, [0, len(t.children) - 1], ["LP", "RP"]) and len(t.children)==2:
            # print(t.children)
            return []

        # e ::= 1234..
        if match(t.children, [0], ["NUMBER"]) and len(t.children) == 1:
            return t

        # e ::= 'ciao'
        if match(t.children, [0, 1, 2], ["SQ", "W", "SQ"]):
            return Token("W", "'" + t.children[1] + "'")

        # e ::= e + e
        if match(t.children, [1], ["PLUS"]) and len(t.children) == 3:
            return Tree(t.data, [self.transform_simple(t.children)])
        # e ::= e - e
        if match(t.children, [1], ["SUB"]) and len(t.children) == 3:
            return Tree(t.data, [self.transform_simple(t.children)])
        # e ::= e * e
        if match(t.children, [1], ["MULT"]) and len(t.children) == 3:
            return Tree(t.data, [self.transform_simple(t.children)])
        # e ::= e / e
        if match(t.children, [1], ["DIV"]) and len(t.children) == 3:
            return Tree(t.data, [self.transform_simple(t.children)])

        if match(t.children, [0], ["WITH"]) and len(t.children) > 1:
            return Tree(t.data, map(self.transform_simple, t.children[1:len(t.children)], "", ","))
        if match(t.children, [0], ["AT"]):
            return Tree(t.data, [t.children[1:]])

        if match(t.children, [0, 1], ["ID", "AR"]) and len(t.children) == 2:
            self.mll.set_current_branch(t.children[0])
            return None
        if match(t.children, [0, 1], ["ID", "EQ"]) and len(t.children) > 2:
            return Tree(t.data, [self.transform_simple(t.children)])

        # significa che in entrata ho solo e+ perciò non ricade in nessun caso
        # se abbiamo e+ l' unica cosa da fare è ,etterci la vitgola in mezzo
        return map(self.transform_simple, t.children, "CO", ",")

    ###################################################################
    #                        SIMPLE MODEL                             #
    ###################################################################

    def traduce_simple_model(self, t: Tree):

        t.children[2:] = self.mll.put_macros(t.children[2:])
        t.children[2:] = apply(t.children[2:], lambda x: x, clean_tok)
        t.children[2:] = self.transform_simple(t.children[2:])

        apply(t, lambda x: x, self.mll.select_imported_libraries)

        self.mll.models[clean_tok(t.children[0]).value] = 0
        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)

        self.mll.ordered_models.append(clean_tok(t.children[0]).value)

        # branches = map(self.transform_simple, t.children[2:], "CO", ",")



        if isTree(t.children[2:][0]) and len(t.children[2:][0].children) == 1:
            return Tree(t.data,
                        [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                         Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "=")]
                        +
                        t.children[2:]
                        + [Token("LP", "("), Token("RP", ")"), Token("WS", "\n")])


        return Tree(t.data,
                    [Token("ID", "models"), Token("LSP", "["), Token("SQ", "'"), clean_tok(t.children[0]),
                     Token("SQ", "'"), Token("LSP", "]"), Token("EQ", "=")]
                    +
                    t.children[2:]
                    + [Token("WS", "\n\n")])