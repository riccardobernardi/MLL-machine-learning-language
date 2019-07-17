import copy

from lark import Token, Tree
from lark.tree import pydot__tree_to_png
from termcolor import cprint

from mll.utils import presentation, clean_tok, get_keras_layers, get_sklearn_models, get_mlxtend_models, stampa, istok, \
    isTree, group, flatten, match, substitute_comp_SQ_W_SQ, map, apply, wellformed, escape


class superMLL:
    def __init__(self, program: str, env={}) -> None:

        # deve essere {"Conv2D" : "from keras.layers import Conv2D"}
        # contains available libraries such as keras, mlxtend, sklearn
        self.string = ""
        self.current_bindings = []
        self.available_libraries = {}

        # contains currently used libraries in the program
        # has to be ["from keras.layers import Conv2D"]
        self.used_libraries = []

        # nomi esterni
        self.env = env

        # modelli da esportare
        self.models = {}
        # modelli ordinati per prendere l' ultimo o il penultimo in ordine di esecuzione
        self.ordered_models = []

        # macros {ID, Tree}
        self.macros = {}

        #parmacs {ID, String 'ID' String}
        self.parmacs = {}

        #non so
        self.import_from_glob = {}

        # the two ASTs
        self.after_tree = None
        self.before_tree = None

        # newer pieces
        self.regressors = []
        self.classifiers = []
        self.current_branch = 0
        self.current_binding_name = None

        self.function_trees = {}

        # MLL program
        self.program = program.replace("with ","")

        self.isInner = False

        # avviso per programma vuoto
        # if self.program.__len__()==0:
        #     cprint("WARNING: your program is empty",'red')

    def select_imported_libraries(self, t: Token) -> None:
        s = clean_tok(t).value
        if s in self.available_libraries.keys():
            if self.available_libraries[s] not in self.used_libraries:
                self.used_libraries.append(self.available_libraries[s])

    def create_available_imports(self):
        imp = get_keras_layers()
        for i in imp.keys():
            for j in imp[i]:
                a = "from keras."+i + " import "+j + "\n"
                self.available_libraries[j] = a

        imp = get_sklearn_models()
        for i in imp.keys():
            for j in imp[i]:
                self.available_libraries[j] = "from sklearn."+i + " import "+j + "\n"

        imp = get_mlxtend_models()
        for i in imp.keys():
            for j in imp[i]:
                self.available_libraries[j] = "from mlxtend."+i + " import "+j + "\n"

    def get_string(self) -> str:
        return self.string

    def last_model(self) -> object:
        return self.models[self.ordered_models[len(self.ordered_models)-1]]

    def print_tree(self):
        stampa(self.after_tree)

    def image_tree(self, which="after"):
        if which == "after":
            pydot__tree_to_png(self.after_tree, "../tree-after.png")
        else:
            if which == "before":
                pydot__tree_to_png(self.after_tree, "../tree-before.png")
            else:
                pydot__tree_to_png(self.after_tree, "../tree-after.png")

    def get_tree_before(self):
        return self.before_tree

    def get_tree_after(self):
        return self.after_tree

    def set_current_branch(self, param):
        self.current_branch = param

    def put_macros(self, t):

        # print(t)

        if isinstance(t, Token):
            if clean_tok(t).value in self.parmacs.keys():
                # print("---before",clean_tok(t).value,"; after",self.parmacs[clean_tok(t).value])
                return self.parmacs[clean_tok(t).value]

            if clean_tok(t).value in self.macros.keys():
                # print("---before", clean_tok(t).value, "; after", self.macros[clean_tok(t).value])
                m = escape(self.macros[clean_tok(t).value])
                return self.put_macros(m)

            if clean_tok(t).value[:3] in self.macros.keys():
                m = self.macros[clean_tok(t).value[:3]]
                m = copy.deepcopy(m)
                m = escape(m)
                export:str = clean_tok(t).value[3:]

                # se quindi c'è solo la grandezza del filtro
                if len(export) == 2:
                    m.children[0].children[1:1] = [Tree("e",[Token("ID",export)])]

                return self.put_macros(m)

            # print("---last return", clean_tok(t).value)
            return clean_tok(t)

        if isinstance(t, Tree):
            return Tree(t.data, self.put_macros(t.children))

        if isinstance(t, list):
            t = escape(t)
            return [self.put_macros(x) for x in t]

        raise Exception("caso inaspettato", t, type(t))

    def solve_macro(self, t):
        return self.macros[clean_tok(t).value]

    def solve_parmac(self, t):
        if clean_tok(t).value in self.parmacs.keys():
            return self.parmacs[clean_tok(t).value]
        else:
            return t

    def add_tab_top_level(self, arr2d: []):
        for i in arr2d:
            i.children.insert(0,Token("WS","\n\t"))

        return arr2d

    def substitute_model(self, t: Token):
        if clean_tok(t).value in self.models.keys():
            return Token("ID","models['"+clean_tok(t).value+"']")
        else:
            return t

    def execute(self):
        s = self.get_string()
        self.env.update({"models":self.models})

        # cprint(self.env.keys(), "green")
        # cprint("inputs" in self.env.keys(),"red")
        # cprint(type(self.env["inputs"]), "green")

        # s = "print('inputs' in locals())\n\nprint('inputs' in globals())\n\n" + s

        # print(s)

        exec(s,self.env)

        # cprint(self.env.keys(), "blue")
        # cprint(self.env["models"], "red")

        return self

    def insert_parmac(self, t: Tree):
        id = t.children[0].value
        a = t.children[2:]
        a = flatten(group(a, "OR"))

        # print(a)

        for i in a:
            self.parmacs[clean_tok(i).value] = Tree("comp", [Token("ID", id), Token("EQ", "="), Token("ID", "'" + i.value + "'")])