from lark import Tree, Token
from termcolor import cprint

from mll.dispatcher import Dispatcher
from mll.utils import apply, clean_tok, group, match, alphabet, isTree, istok, filter, map, escape, list_types_list


class ForkedModel:

    def __init__(self, mll):
        self.mll = mll

    ###################################################################
    #                           DAG MODEL                             #
    ###################################################################

    def traduce_forks(self, t: Tree):

        # non esistono i modelli che sono funzioni quindi non si mette mai la riga sotto, salverò in un altro luogo
        # self.mll.models[clean_tok(t.children[0]).value] = t
        # esempio

        # cprint("ATTENTION FORK","red")
        self.mll.function_trees[clean_tok(t.children[0]).value] = t

        t.children[2:] = escape(t.children[2:])
        t.children[2:] = self.mll.put_macros(t.children[2:])
        t.children[2:] = apply(t.children[2:], lambda x:x, clean_tok )

        # print("FORKED tree after putmacros:",t)

        # t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.solve_parmac)
        # t = apply(t, lambda x: x, clean_tok)
        apply(t, lambda x: x, self.mll.select_imported_libraries)

        t.children[2:] = apply(t.children[2:], lambda x: x, self.mll.substitute_model)
        # t = apply(t, lambda x: x, self.mll.solve_parmac)

        branches = t.children[2:]
        branches = list(group(branches, "PI"))

        a = []
        for branch in branches:
            a = a + self.traduce_branch(branch)

        branches = a

        b = self.mll.current_bindings[len(self.mll.current_bindings)-1]

        #  resetto tutto quando faccio il return
        self.mll.current_bindings = []
        self.mll.current_branch = 0

        return Tree(t.data,
                    [
                        Token("ID", "def "), Token("ID", clean_tok(t.children[0]).value), Token("ID", "(x)"), Token("COLON", ":")
                    ] +
                    self.mll.add_tab_top_level(filter(lambda x: x is not None, branches)) +
                    [
                        Token("ID", "return "), Token("ID", b), Token("WS", "\n\n")
                    ]
                    )

    def traduce_branch(self, branch: list) -> list:

        # cprint(branch,"yellow")
        # branch contiene una e perciò la facciamo uscire
        a = []
        for i in branch:
            self.mll.current_branch += 1
            self.mll.current_bindings.append(alphabet[self.mll.current_branch - 1])
            a.append(self.traduce_layers(i, "first"))
        return a

    # All' inizio t è una lista ma tutte le successive call sono su trees
    def traduce_layers(self, t: Tree, opt=""):
        # (layer + ( layer + layer ))

        if isinstance(t,list):
            t=t[0]

        # cprint("before", "blue")
        # print(t.children)
        # print(list_types_list(t.children))
        # print(len(t.children))

        #####################################################################
        #           e + e
        #####################################################################

        if match(t.children, [1], ["PLUS"]) and len(t.children) == 3:
            if opt == "first":
                # print("fsx:", t.children[0])
                # print("fdx:", t.children[2])
                return Tree(t.data, [Tree(t.children[0].data, [self.traduce_layers(t.children[0], "first")]),
                                     Tree(t.children[2].data, [self.traduce_layers(t.children[2])])])
            else:
                # print("sx:", t.children[0])
                # print("dx:", t.children[2])
                return Tree(t.data, [Tree(t.children[0].data, [self.traduce_layers(t.children[0])]),
                                     Tree(t.children[2].data, [self.traduce_layers(t.children[2])])])

        #####################################################################
        #           ( [e]+ )
        #####################################################################

        # e ::= LP e RP
        if match(t.children, [0, len(t.children) - 1], ["LP", "RP"]):
            return Tree(t.data, [Token("ID", alphabet[self.mll.current_branch - 1]), Token("EQ", "="),
                                 Token("LP", "(")] +
                        [Dispatcher(self.mll,"forked").transform(t.children[1])] +[Token("RP", ")"), Token("LP", "("), Token("ID", alphabet[self.mll.current_branch - 1]), Token("RP", ")"),
                            Token("WS", "\n\t")])

        #####################################################################
        #           ID ->
        #####################################################################

        if match(t.children, [0,1], ["ID","AR"]) and len(t.children) == 2:
            # preparati per dare 2 bindings
            # print("sono della ID AR len2")
            self.mll.current_branch -=1
            self.mll.current_bindings = self.mll.current_bindings[:len(self.mll.current_bindings)-1]
            self.mll.current_binding_name = t.children[0]
            return None

        #####################################################################
        #           ID -> ID ID
        #####################################################################

        if match(t.children, [0,1,2,3], ["ID","AR","ID","ID"]) and len(t.children) == 4:
            # concatena 2 concatenazioni che sono state bindate
            # print("sono della ID AR ID ID len4")
            return Tree(t.data,
                        [
                            Token("ID", alphabet[self.mll.current_branch - 1]),
                            Token("EQ", "="), t.children[0],
                            Token("LP", "("),
                            Token("RP", ")"),
                            Token("LP", "("),
                            Token("LSP", "["),
                            Token("ID", clean_tok(t.children[2]).value),
                            Token("CO", ","),
                            Token("ID", clean_tok(t.children[3]).value),
                            Token("RSP", "]"),
                            Token("RP", ")"),
                            Token("WS", "\n\t")
                        ])

        #####################################################################
        #           e, e, e
        #####################################################################

        t.children[1:] = map(Dispatcher(self.mll,"forked").transform, t.children[1:], "CO", ",")

        #####################################################################
        #           ID e+ w/o "+" and "->"
        #####################################################################

        if match(t.children, [0], ["ID"]) and len(t.children) > 1 and not match(t.children, [], ["PLUS"]) and not match(t.children, [], ["AR"]):
            if clean_tok(t.children[0]).value == "assign":
                return Tree(t.data, [Token("ID", alphabet[self.mll.current_branch - 1]), Token("EQ", "="), t.children[0],
                                     Token("LP", "(")] + t.children[1:] + [
                                Token("RP", ")"),
                                Token("WS", "\n\t")])

            if opt == "first":
                return Tree(t.data, [Token("ID", alphabet[self.mll.current_branch - 1]), Token("EQ", "="), t.children[0],
                                     Token("LP", "(")] + t.children[1:] + [
                                Token("RP", ")"), Token("LP", "("), Token("ID", "x"), Token("RP", ")"),
                                Token("WS", "\n\t")])
            else:
                return Tree(t.data,
                            [
                                Token("ID", alphabet[self.mll.current_branch - 1]),
                                Token("EQ", "="), t.children[0],
                                Token("LP", "(")] +
                            t.children[1:] +
                            [
                                Token("RP", ")"),
                                Token("LP", "("),
                                Token("ID", alphabet[self.mll.current_branch - 1]),
                                Token("RP", ")"),
                                Token("WS", "\n\t")
                            ])

        ###############################################################################
        #           ID and no binding for forks and name is operations and not alias
        ###############################################################################

        if match(t.children, [0], ["ID"]) and len(t.children) == 1 and self.mll.current_binding_name is not None and clean_tok(t.children[0]).value not in self.mll.models:
            a = str(self.mll.current_bindings[:len(self.mll.current_bindings) - 1]).replace("'", "").replace("x,", "")
            # è sempre x il branch corente perchè vogliamo che il branch bindato sia stealth
            self.mll.current_bindings = ["x"]

            return Tree(t.data,
                        [
                            Token("ID", clean_tok(self.mll.current_binding_name)),
                            Token("EQ", "="), t.children[0],
                            Token("LP", "("),
                            Token("RP", ")"),
                            Token("LP", "("),
                            Token("ID", a),
                            Token("RP", ")"),
                            Token("WS", "\n\t")
                        ])

        #####################################################################
        #           ID and binding for forks exists and name is alias
        #####################################################################

        # print(match(t.children, [0], ["ID"]))
        # print(len(t.children) == 1)
        # print(self.mll.current_binding_name is not None)
        # print(clean_tok(t.children[0]).value in self.mll.models if isinstance(t.children[0],Token) else "Tree")

        if match(t.children, [0], ["ID"]) and len(t.children) == 1 and self.mll.current_binding_name is not None and clean_tok(t.children[0]).value.replace("models['","").replace("']","") in self.mll.models.keys():
            # a = str(self.mll.current_bindings[:len(self.mll.current_bindings) - 1]).replace("'", "").replace("x,", "")
            # è sempre x il branch corente perchè vogliamo che il branch bindato sia stealth

            return Tree(t.data,
                        [
                            Token("ID", clean_tok(alphabet[self.mll.current_branch - 1])),
                            Token("EQ", "="), t.children[0]
                        ])

        # print("entro nel secondo lock", t.children)
        # print(match(t.children, [0], ["ID"]))
        # print(len(t.children) == 1)
        # print(self.mll.current_binding_name is None)
        # print(clean_tok(t.children[0]).value in self.mll.models if isinstance(t.children[0], Token) else "Tree")

        #####################################################################
        #           ID and binding for forks non-exists and name is alias
        #####################################################################

        if match(t.children, [0], ["ID"]) and len(t.children) == 1 and self.mll.current_binding_name is None and clean_tok(t.children[0]).value.replace("models['","").replace("']","") in self.mll.models.keys():
            # a = str(self.mll.current_bindings[:len(self.mll.current_bindings) - 1]).replace("'", "").replace("x,", "")
            # è sempre x il branch corente perchè vogliamo che il branch bindato sia stealth

            return Tree(t.data,
                        [
                            Token("ID", alphabet[self.mll.current_branch - 1]),
                            Token("EQ", "="), t.children[0], Token("WS", "\n\t")
                        ])

        #####################################################################
        #           ID and no binding name for forks
        #####################################################################

        if match(t.children, [0], ["ID"]) and len(t.children) == 1 and self.mll.current_binding_name is None:
            a = str(self.mll.current_bindings[:len(self.mll.current_bindings) - 1]).replace("'", "").replace("x,", "")
            self.mll.current_bindings = ["x"]

            return Tree(t.data,
                        [
                            Token("ID", "x"),
                            Token("EQ", "="), t.children[0],
                            Token("LP", "("),
                            Token("RP", ")"),
                            Token("LP", "("),
                            Token("ID", a),
                            Token("RP", ")"),
                            Token("WS", "\n\t")
                        ])

        #####################################################################
        #           AT e become a=f()(a)
        #####################################################################

        if match(t.children, [0], ["AT"]):
            return Tree(t.data, [Token("ID", alphabet[self.mll.current_branch - 1]), Token("EQ", "="), t.children[1],
                                 Token("LP", "("),
                            Token("RP", ")"), Token("LP", "("), Token("ID", alphabet[self.mll.current_branch - 1]), Token("RP", ")"),
                            Token("WS", "\n\t")])

        # cprint("after", "red")
        # print(t.children)
        # print(len(t.children))
        # print(list_types_list(t.children))
        # print("-------------------------------------")

        return Tree(t.data, [self.traduce_layers(t.children, opt)])