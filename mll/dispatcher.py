from lark import Token, Tree

from mll.mlltranspiler import MLL
from mll.utils import match

from mll.utils import apply, clean_tok, group, match, alphabet, isTree, istok, filter, map, escape, list_types_list

class Dispatcher:

    def __init__(self,mll:MLL,model_type="simple"):
        self.model_type = model_type
        self.mll = mll

    ###################################################################
    #                        DISPATCHER                               #
    ###################################################################

    def transform(self, t: object):
        if isinstance(t, Token):
            return self.translate_token(t)
        elif isinstance(t, Tree):
            return self.translate_tree(t)
        elif isinstance(t, type([])):
            return self.translate_list(t)
        else:
            raise Exception("Non esiste questo caso nella fun transform: ", type(t))

    def translate_tree(self, t: Tree) -> Tree:
        if t.data == "comp":
            return Tree(t.data, self.transform(t.children))

        if t.data == "e" or t.data == "macro":
            return self.translate_e(t)

        return t

    def translate_token(self, t: Token) -> object:
        t = self.mll.put_macros(t)

        self.mll.select_imported_libraries(t)

        return t

    def translate_list(self, t: list):
        return filter(lambda x: x is not None, [self.transform(x) for x in t])

    def translate_e(self, t: Tree):
        # print(list_types_list(t.children))

        # e ::= ID
        if match(t.children, [0], ["ID"]) and len(t.children) == 1:
            return Tree(t.data, self.transform(t.children))
        # e ::= ID e+
        if match(t.children, [0], ["ID"]) and len(t.children) > 1 and not match(t.children, [], ["PLUS"]):
            return Tree(t.data,
                        [t.children[0], Token("LP", "(")] +
                        map(self.transform, t.children[1:len(t.children)], "CO", ",") +
                        [Token("RP", ")")])

        # e ::= AT ID LP RP
        if match(t.children, [0, 1, 2, 3], ["AT", "ID", "LP", "RP"]):
            return Tree(t.data, t.children[1:])

        # e ::= LP e RP
        if match(t.children, [0, 2], ["LP", "RP"]):
            return Tree(t.data,
                        [self.transform(t.children[1])])

        # e ::= 1234..
        if match(t.children, [0], ["NUMBER"]) and len(t.children) == 1:
            return t

        # e ::= 'ciao'
        if match(t.children, [0, 1, 2], ["SQ", "W", "SQ"]):
            return Tree(t.data,[Token("W", "'" + t.children[1] + "'")])





        if self.model_type == "forked" or self.model_type =="simple":

            # e ::= e + e
            if match(t.children, [1], ["PLUS"]) and len(t.children) == 3:
                return Tree(t.data, self.transform(t.children))

        if self.model_type == "sequential":

            # e ::= e + e
            if match(t.children, [1], ["PLUS"]) and len(t.children) == 3:
                return Tree(t.data, [self.transform(t.children[0]), Token("CO", ","),
                                     self.transform(t.children[2])])





        # e ::= e - e
        if match(t.children, [1], ["SUB"]) and len(t.children) == 3:
            return Tree(t.data, self.transform(t.children))
        # e ::= e * e
        if match(t.children, [1], ["MULT"]) and len(t.children) == 3:
            return Tree(t.data, self.transform(t.children))
        # e ::= e / e
        if match(t.children, [1], ["DIV"]) and len(t.children) == 3:
            return Tree(t.data, self.transform(t.children))

        if match(t.children, [0], ["WITH"]) and len(t.children) > 1:
            return Tree(t.data, map(self.transform, t.children[1:len(t.children)], "", ","))
        if match(t.children, [0], ["AT"]):
            return Tree(t.data, t.children[1:] + [Token("LP", "("), Token("RP", ")")])

        if match(t.children, [0, 1], ["ID", "AR"]) and len(t.children) == 2:
            self.mll.set_current_branch(t.children[0])
            return None
        if match(t.children, [0, 1], ["ID", "EQ"]) and len(t.children) > 2:
            return Tree(t.data, self.transform(t.children))

        return Tree(t.data, map(self.transform, t.children, "CO", ","))