from lark import Lark, Token
from lark.tree import Tree, pydot__tree_to_png
from termcolor import cprint

from mll.new_grammar import get_new_grammar
from mll.utils import scrivi, istok, clean_tok, plus_in_array, clean_deep, clean_arr, escape, \
    get_keras_layers, uncomma, isTree, presentation, clean_tabs, get_sklearn_models, get_utils_functions, \
    get_base_imports, get_mlxtend_models, remove_AT, stampa

import warnings
warnings.filterwarnings("ignore")


class MLL:

    def __init__(self,program: str,loc = {}) -> None:
        presentation()
        self.loc = loc
        self.param_values = {}
        self.models = {}
        self.macros={}
        self.ordered_models = []
        self.current_keras = None
        self.actual_imports = ""
        self.actual_imports_set = set()
        self.model_type = {} #model_name : regressor|classifier
        self.import_from_glob = {}
        self.after_tree = None
        self.before_tree = None

        self.program = program

        if self.program.__len__()==0:
            cprint("WARNING: your program is empty",'red')

    def start(self):
        self.string = uncomma(self.transpile(self.program))

    def is_in_possible_imports(self, tok:str) -> str :
        for i in get_keras_layers().keys():
            if tok in get_keras_layers()[i]:
                return "from keras."+i + " import "+tok + "\n"

        for i in get_sklearn_models().keys():
            if tok in get_sklearn_models()[i]:
                return "from sklearn."+i + " import "+tok + "\n"

        for i in get_mlxtend_models().keys():
            if tok in get_mlxtend_models()[i]:
                return "from mlxtend."+i + " import "+tok + "\n"

        return ""

    def concat_array(self,t: list) -> list:
        ok = False
        i = 0
        opened = False
        while True:
            if i == len(t):
                break
            else:
                if istok(t[i]) and clean_tok(t[i].value) == "concatenate":
                    ok = True

                if ok and istok(t[i]) and clean_tok(t[i].value) == "(" and i+1 < len(t) and (not istok(t[i + 1]) or not clean_tok(t[i + 1].value) == "["):
                    t.insert(i+1, Token("SP","[") )
                    opened = True

                if ok and opened and istok(t[i]) and clean_tok(t[i].value) == ")" and i==(len(t) - 1):
                    t.insert(i , Token("SP","]") )
                    ok = False
                    opened = False

                i += 1
        return t

    def macro_operations(self, t: list) -> Token:

        m = clean_tok(t[0].value)

        t=t[2:]

        #devo fare l' escape da sx a dx per relu tree che diventa relu plain token
        t = escape(m, t)

        self.macros[m] = t

        return Token("DELETED", "#macro: " + scrivi(t).replace("\n", "") + "\n")

    def recon_class_ids(self, t:object) -> None:
        if isinstance(t, Token):
            if t.type == "ID":
                if clean_tok(t.value) not in self.actual_imports_set and self.is_in_possible_imports(clean_tok(t.value)) != "":
                    self.actual_imports += self.is_in_possible_imports(clean_tok(t.value))
                    self.actual_imports_set.add(clean_tok(t.value))

            if t.type == "CONCAT":
                self.actual_imports += self.is_in_possible_imports(clean_tok("concatenate"))
                self.actual_imports_set.add(clean_tok("concatenate"))

            if (t.type == "ID" or t.type == "FEXTNAME") and "@" in t.value:
                locals().update(self.loc)
                try:
                    self.import_from_glob[str(clean_tok(t.value)).replace("@","")] = locals()[
                        str(clean_tok(t.value)).replace("@","")]
                except:
                    try:
                        self.import_from_glob[str(clean_tok(t.value)).replace("@", "")] = globals()[
                            str(clean_tok(t.value)).replace("@", "")]
                    except:
                        print("la variabile "+ str(clean_tok(t.value)).replace("@", "") +" non è disponibile")

        elif isinstance(t, Tree):
            self.recon_class_ids(t.children)
        elif isinstance(t, list):
            for i in t:
                self.recon_class_ids(i)
        else:
            raise Exception("Non esiste questo caso nella fun clean_deep")

    def format_commas(self,t: list) -> list:

        flag = False

        for i in t:
            if flag == True and ( (isinstance(i, Tree) and (i.data == "e" or i.data == "comp" )) or (isinstance(i,Token) and i.type == "WITH") ):
                if isinstance(i, Tree):
                    i.children.insert(0, Token("", ","))
                    flag = True
                if isinstance(i,Token):
                    index = t.index(i)
                    t.remove(i)
                    t.insert(index,Token("CO",","))
            else:
                if flag == False and (isinstance(i, Tree) and (i.data == "e" or i.data == "comp" ) or (isinstance(i,Token) and i.type == "WITH") ):
                    flag = True
                else:
                    flag = False
        return t

    def format_parenthesis(self,t: list) -> list:

        for i in t:
            if istok(i) and plus_in_array(t):
                i.value = clean_tok(i)

        if t[1].type == "COLON" :
            t[1].value = "="

            if clean_tok(t[2].value) not in self.models.keys():
                t.insert(3, Token("P", "("))

                i = 0
                while True:
                    if i == len(t):
                        break
                    else:
                        if istok(t[i]) and clean_tok(t[i].value) == "+":
                            t.insert(i , Token("PP", ")\n"))
                            break

                        i += 1
            else:
                t.insert(3, Token("P", "\n"))

        opened = False

        i = 0
        while True:
            if i == len(t):
                break
            else:
                if istok(t[i]) and clean_tok(t[i].value) == "+" and opened:
                    t.insert(i, Token("P", "))\n"))
                    opened = False
                    i += 1
                if istok(t[i]) and clean_tok(t[i].value)== "+":
                    t.pop(i)
                    t.insert(i,Token("ID",self.current_keras))
                    t.insert(i+1,Token("ADD",".add("))
                    opened = True
                    t.insert(i + 3, Token("P", "("))
                    i+=3
                i += 1

        t.append(Token("PP", "))\n"))

        return t

    def put_macros(self,t: list) -> list:

        for i in range(0, len(t)):

            ok=False

            if istok(t[i]):
                for j in self.macros.keys():
                    if clean_tok(t[i].value) == j :
                        ok = True
                        break

                if ok:
                    s = clean_tok(t[i].value)
                    # pulisco la parole nel token dell list che deve essere sostituita dalla macro
                    # prendo l' index della parola da sostituire
                    # rimuovo la parola da sostituire
                    # inserisco al posto della parola un intero list che contiene la macro

                    t.pop(i)

                    #una macro può contenere macro a sua volta
                    self.macros[s] = self.put_macros(self.macros[s])

                    #metto la macro
                    t[i:i] = self.macros[s]
                    i += self.macros[s].__len__()

        return t

    def format_keras(self,t: list) -> list:
        #ricevo un list di elementi che è figlio di un model
        #il model era un keras-model perciò devo formattarlo correttamente

        if plus_in_array(t):

            self.ordered_models.append(clean_tok(t[0].value))
            self.models[clean_tok(t[0].value)] = 0

            self.current_keras:str = clean_tok(t[0].value)
            t[1].value = "="

            #cerco la sostituzione di macros per sostituire seq con Sequential
            for i in t:
                for j in self.macros.keys():
                    if istok(i) and clean_tok(i) == j:
                        t = self.put_macros(t)

            t = self.format_parenthesis(t)

            t = self.format_commas(t)

        else:

            self.ordered_models.append(clean_tok(t[0].value))
            self.models[clean_tok(t[0].value)] = 0

            t = self.format_parens(t)

        return t

    def format_parens(self,t: list) -> list:

        for i in range(0,len(t)):

            if istok(t[i]) and t[i].type == "COLON":
                t[i].value = "="

            if (istok(t[i]) and t[i].type== "ID" and i-1>0 and istok(t[i - 1]) and t[i - 1].type== "COLON") :
                t[i].value = clean_tok(t[i].value)

                t.insert(i+1,Token("P","("))

                #####dovrei mettere len(t)-1 ma lui non ricalcola la lunghezza corretta, colpa di python!!!
                t.insert(len(t) , Token("P", ")\n"))

        return t

    def save_parmac(self,t: list) -> list:
        #t è un array di id ed e che costituisce una macro inversa

        param_name = t[0].value
        t = t[1:]

        arr = []

        for i in t:
            if isTree(i) and i.data == "n":
                self.param_values[i.children[1].value] = param_name + "=" + scrivi(i).replace("\n","")
                arr += [Token("DELETED","# parmac: "+scrivi(i).replace("\n","")+"\n")]

        return arr

    def dag(self,t:list) -> list:

        t = self.put_macros(t)
        t = clean_arr(t)

        names = ['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t','u','v','z']
        models = []
        to_concat_and_free = []
        concat_in_array = False

        t.pop(1)

        t.insert(0, Token("DEF", "\ndef "))
        val = t[1].value + "(x):"
        t.pop(1)
        t.insert(1, Token("ID", val))
        t.insert(2, Token("TAB","\n\t"))

        models.insert(0, names.pop(0))
        to_concat_and_free += [models[0]]

        t.pop(3)
        t.insert(3, Token("ASSIGN", models[0] + "=("))
        t.insert(5,Token("P","("))

        first = True
        t.append(Token("PIPE","|"))

        for i in t:
            if istok(i) and clean_tok(i.value)=="concat":
                concat_in_array = True

        concatenated = False
        model_x = "x"
        skip = False

        i = 0
        while True:
            if i == len(t):
                break
            else:
                t = self.put_macros(t)

                if istok(t[i]) and clean_tok(t[i].value) == "assign":
                    s = clean_tok(t[i+2].children[0].value)
                    t.pop(i+2)
                    t.insert(i+2,Token("ASSIGN","models['"+s+"']"))
                    skip = True

                if istok(t[i]) and clean_tok(t[i].value) == "concat":

                    names.insert(0,models[0])
                    models.pop(0)
                    to_concat_and_free = to_concat_and_free[:len(to_concat_and_free)-1]

                    t.pop(i)
                    t.pop(i)
                    t.pop(i - 1)

                    if len(to_concat_and_free) >1:

                        t.insert(i,Token("CONCAT",models[0] + " = concatenate(["))
                        model_x = models[0]

                        s = ""

                        for j in to_concat_and_free:
                            if to_concat_and_free.index(j) < (len(to_concat_and_free) - 1):
                                s += str(j)
                                s += ","
                            else:
                                s += str(j)

                        t.insert(i+1, Token("MODELS", s))
                        t.insert(i+2, Token("PP","])"))

                        to_concat_and_free = []

                        models.insert(0, names.pop(0))
                        to_concat_and_free += [models[0]]

                        t.insert(i + 3, Token("TAB", "\n\t"))

                        t.insert(i + 4, Token("ASSIGN", models[0] + "=("))
                        t.insert(i + 6, Token("P", "("))

                        t.pop(i-1)

                        concatenated = True

                    else:

                        to_concat_and_free = []

                        models.insert(0, names.pop(0))

                        t.insert(i, Token("TAB", "\n\t"))

                        t.insert(i + 1, Token("ASSIGN", models[0] + "=("))
                        t.insert(i + 3, Token("P", "("))

                        t.pop(i - 1)

                if istok(t[i]) and (clean_tok(t[i].value) == "+" or clean_tok(t[i].value) == "|") and first:
                    if clean_tok(t[i].value) == "|":
                        first = True
                        models.insert(0, names.pop(0))
                        to_concat_and_free += [models[0]]
                    else:
                        first = False
                    t.pop(i)

                    if not skip:
                        if concatenated == False:
                            t.insert(i,Token("MODEL","))("+ model_x +")"))
                        else:
                            t.insert(i,Token("MODEL","))("+model_x+")"))
                            concatenated = False
                    else:
                        t.insert(i, Token("PP", "))"))
                        skip = False

                    t.insert(i + 1, Token("TAB", "\n\t"))

                    t.insert(i + 2, Token("ASSIGN", models[0] + "=("))
                    t.insert(i + 4, Token("P", "("))

                if istok(t[i]) and (clean_tok(t[i].value) == "+" or clean_tok(t[i].value) == "|") and not first:
                    if clean_tok(t[i].value) == "|":
                        first = True
                        models.insert(0, names.pop(0))
                        t.insert(i, Token("MODEL", "))(" + models[1] + ")"))
                        to_concat_and_free += [models[0]]
                    else:
                        t.insert(i, Token("MODEL", "))(" + models[0] + ")"))
                        first = False

                    t.pop(i+1)

                    t.insert(i + 1, Token("TAB", "\n\t"))

                    t.insert(i + 2, Token("ASSIGN", models[0] + "=("))
                    t.insert(i + 4, Token("P", "("))

                i += 1

        t = t[:len(t) - 2]
        models.pop(0)

        if len(models) > 1 and concat_in_array == False:

            s = []

            for i in models:
                if models.index(i) < (len(models) - 1):
                    s += [i]
                    s += [","]
                else:
                    s += [i]

            t.append(Token("CONCAT", "x = concatenate(["))
            models.insert(0,"x")
            t.append( [Token("MODELS", i) for i in s ])
            t.append( Token("PP","])\n\t") )

        t.append( Token("RETURN", "return " + models[0] + "\n\n"))

        t = self.format_commas(t)

        t = clean_deep(t)

        return t

    def transform(self,t : object) -> object:
        if isinstance(t, Token):

            s = t.value.replace(" ","")
            l = t.type

            if not(l == "PP" or l == "P"):
                s = s.replace("\n","")

            if "\n" in s:
                if s.replace("\n","") in self.models.keys() and ("models['"+s.replace("\n","")+"']") not in s:
                    s = ("models['"+s.replace("\n","")+"']\n")
            else:
                if s in self.models.keys() and ("models['"+s+"']") not in s:
                    s = ("models['"+s.replace("\n","")+"']")

            if "@" in s:
                s = s.replace("@", "")

            return Token(l, s)

        elif isinstance(t, Tree):

            # if t.data == "comment":
            #     return Token("comment", scrivi(t.children).replace("\n","").replace("\t","").replace("\r",""))

            if t.data == "fapp":
                self.recon_class_ids(t.children)
                return Token("fapp", scrivi(t.children))

            if t.data == "dotname":
                return Token("ID",scrivi(t.children))

            if t.data == "dag":
                self.recon_class_ids(t.children)
                t.children = remove_AT(t.children)
                return Tree(t.data, self.dag(t.children))

            if t.data == "macro":
                self.recon_class_ids(t.children)
                t.children = remove_AT(t.children)
                return self.macro_operations(t.children)

            if t.data == "comment":
                return clean_tabs(t)

            if t.data == "model":

                if isTree(t.children[0]) and t.children[0].data == "mt" and (clean_tok(t.children[0].children[0].value.lower()) == "regressor" or clean_tok(t.children[0].children[0].value.lower()) == "classifier"):
                    self.model_type[clean_tok(t.children[1].value)] = clean_tok(t.children[0].children[0].value.lower())
                    #salvo l' informazione del fatto che un modello è regressor o classifier e poi elimino questa info per retrocompatibility
                    t.children = t.children[1:]

                self.recon_class_ids(t.children)

                t.children = self.format_keras(t.children)

            if t.data == "parmac":
                return self.save_parmac(t.children)

            return Tree(t.data, self.transform(t.children))

        elif isinstance(t, list):

            for i in t:
                if istok(i) and clean_tok(i.value) in self.param_values:
                    i.value = self.param_values[clean_tok(i.value)]
                    i.type = "e" #così gli viene messa correttamente la virgola

            t = clean_arr(t)

            for i in t:
                for j in self.macros.keys():
                    if istok(i) and clean_tok(i.value) == j:
                        t = self.put_macros(t)

            t = self.format_commas(t)

            t = self.concat_array(t)

            return [self.transform(m) for m in t]
        else:
            raise Exception("Non esiste questo caso nella fun transform")

    def transpile(self,program: str) -> str:

        parser = Lark(get_new_grammar(), start='mll')

        self.before_tree = parser.parse(program)

        self.after_tree = Tree(self.before_tree.data, self.transform(self.before_tree.children))
        self.recon_class_ids(self.after_tree)

        s = get_base_imports() + self.actual_imports + get_utils_functions() + scrivi(self.after_tree)

        return s

    def get_string(self) -> str:
        return self.string

    def last_model(self) -> object:
        return self.models[self.ordered_models[len(self.ordered_models)-1]]

    def execute(self):
        s = self.get_string()
        glob = {"models":self.models}
        glob.update(self.import_from_glob)
        exec(s,glob)

    def get_imports(self):
        print(self.actual_imports)

    def print_tree(self):
        stampa(self.after_tree)

    def image_tree(self, which="after"):
        if which == "after":
            pydot__tree_to_png(self.after_tree, "tree-after.png")
        else:
            if which == "before":
                pydot__tree_to_png(self.after_tree, "tree-before.png")
            else:
                pydot__tree_to_png(self.after_tree, "tree-after.png")

    def get_tree_before(self):
        return self.before_tree

    def get_tree_after(self):
        return self.after_tree
