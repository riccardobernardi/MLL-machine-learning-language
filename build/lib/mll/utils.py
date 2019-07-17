from keras import layers, backend
from lark import Tree, Token
from termcolor import cprint


def stampa(t: object) -> None:
    # stampa un albero a schermo, non scrive su stringhe

    if isinstance(t, Token):
        print(t.value, end='')
    elif isinstance(t, Tree):
        stampa(t.children)
    elif isinstance(t, type([])):
        for i in t:
            stampa(i)
    else:
        raise Exception("Non esiste questo caso nella fun stampa")


def scrivi(t: object) -> str:
    # ritorna una stringa con l' albero dentro

    if isinstance(t, Token):
        return t

    if isinstance(t, Tree):
        return scrivi(t.children)

    if isinstance(t, type([])):
        s = ""
        for i in t:
            s += scrivi(i)
        return s

    if isinstance(t, type("")):
        return t

    if t is None:
        return ""

    # cprint(type(t),"blue")
    # print(t)
    raise Exception("Non esiste questo caso nella fun scrivi")


def istok(i: object) -> bool:
    if isinstance(i, Token):
        return True
    else:
        return False


def clean_tok(tok: Token) -> Token:
    # if tok.type=="W":
    #     return tok
    # if tok.type == "WSP":
    #     return Token(tok.type, tok.value.replace("\t", ""))
    return Token(tok.type, tok.value.replace(" ", "").replace("\t", "").replace("\n", ""))


from keras import models


def get_keras_layers() -> dict:
    keras_layers = {}

    keras_layers["models"] = set()
    for k in models.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["models"].add(k)

    keras_layers["layers"] = set()
    for k in layers.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["layers"].add(k)

    keras_layers["backend"] = set()
    for k in backend.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["backend"].add(k)

    return keras_layers


from sklearn import linear_model
from sklearn import cluster
from sklearn import ensemble
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import pipeline


def get_sklearn_models() -> dict:
    keras_layers = {}

    # keras_layers = {n*set()}

    keras_layers["linear_model"] = set()
    for k in linear_model.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["linear_model"].add(k)

    keras_layers["cluster"] = set()
    for k in cluster.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["cluster"].add(k)

    keras_layers["ensemble"] = set()
    for k in ensemble.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["ensemble"].add(k)

    keras_layers["pipeline"] = set()
    for k in pipeline.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["pipeline"].add(k)

    # print("EUREKAAAAAAA","RandomForestClassifier" in keras_layers["ensemble"])

    keras_layers["neighbors"] = set()
    for k in neighbors.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["neighbors"].add(k)

    keras_layers["preprocessing"] = set()
    for k in preprocessing.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["preprocessing"].add(k)

    keras_layers["decomposition"] = set()
    for k in decomposition.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["decomposition"].add(k)

    keras_layers["svm"] = set()
    for k in svm.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["svm"].add(k)

    keras_layers["tree"] = set()
    for k in tree.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["tree"].add(k)

    return keras_layers


from mlxtend import classifier


def get_mlxtend_models() -> dict:
    keras_layers = {}

    # keras_layers = {n*set()}

    keras_layers["classifier"] = set()
    for k in classifier.__dict__.keys():
        if "__" not in k and k != "K":
            keras_layers["classifier"].add(k)

    return keras_layers


def isTree(t: object) -> bool:
    if isinstance(t, Tree):
        return True
    return False


def presentation():
    print()
    print("#################################################################################")
    print("#                            STARTING MLL                                       #")
    print("#################################################################################")
    print("# 1. model = MLL(PROGRAM)                                                       #")
    print("# 2. model.start() -> to pass from MLL to python                                #")
    print("# 3. model.get_string() -> to get python code of your program                   #")
    print("# 4. model.execute() -> to run python code of your program                      #")
    print("# 5. clf = model.last_model() -> to get last model of your program              #")
    print("# 6. MLL() -> to get this window                                                #")
    print("#                                                                               #")
    print("#                                    thesis: MLL-machine learning language      #")
    print("#                                    student: Bernardi Riccardo                 #")
    print("#                                    supervisor: Lucchese Claudio               #")
    print("#                                    co-supervisor: Spanò Alvise                #")
    print("#################################################################################")


def map(f: type(map), arr: [], type_between: type("") = "", between: type("") = "", opt: str = "") -> []:
    a = []
    for i in arr:
        if opt != "":
            a.append(f(i, opt))
        else:
            a.append(f(i))
        if type_between != "":
            a.append(Token(type_between, between))

    if type_between != "":
        a = a[0:len(a) - 1]

    return a


def AND(a, b):
    return a and b


def reduce(f: type(map), arr: []):
    if len(arr) == 1:
        return arr[0]
    else:
        n = f(arr[0], arr[1])
        arr.pop(0)
        arr.pop(0)
        arr.insert(0, n)
        return reduce(f, arr)


def OR(a, b):
    return a or b


def match(children, indexes, tok_types):
    # print(len(indexes))

    try:
        if len(indexes) > 0:
            true = [children[indexes[i]].type == tok_types[i] for i in range(0, len(indexes))]
        else:
            true = []
            for i in range(0, len(children)):
                if isinstance(children[i], Token):
                    true.append(children[i].type == tok_types[0])
    except:
        # cprint("esco sull except della match","green")
        return False

    return reduce(AND, true) if len(indexes) > 0 else reduce(OR, true)


def split(arr: [], split_token: type("")) -> []:
    a = []
    # arr = arr[1:] #tolgo la prima pipe per comodità
    last_split = 0

    for i in range(0, len(arr)):
        if isinstance(arr[i], Token) and arr[i].type == split_token and arr[i] is not None:
            b = arr[last_split:i]
            a.append(b)
            last_split = i + 1

    return a


def flatten(ndim):
    a = []

    for i in ndim:
        if type(i) == list:
            b = flatten(i)
            a = a + b
        else:
            a = a + [i]

    return a


def filter(f, arr: []) -> []:
    a = []

    for i in arr:
        if f(i):
            a.append(i)

    return a


def group(seq: [], sep: str):
    g = []
    for el in seq:
        if type(el) == Token and el.type == sep:
            yield g
            g = []
        if type(el) == Token and el.type == sep:
            continue
        else:
            g = g + [el]
    yield g


# put_tabs = lambda x: Token(x.type, "\t" + x.value) if type(x) == Token else map(put_tabs, x.children)
#
# cut_tabs = lambda x: clean_tok(x) if type(x) == Token else map(cut_tabs, x.children)


def MAX(a: int, b: int):
    return a if a > b else b


def tree_depth(t) -> int:
    if type(t) == Tree:
        # print("from TRee of treedepth",t)
        a: int = tree_depth(t.children)
        # print("from tredepth what come from depthiness",a)
        return 1 + a
    if type(t) == Token:
        return 0
    if isinstance(t, list):
        # print(reduce(MAX, map(tree_depth,t)))
        return reduce(MAX, map(tree_depth, t))

    raise Exception("Errorfffffffff")


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'z']


def list_types_list(l: []) -> []:
    return [x.data if type(x) == Tree else x.type for x in l]


# def escape(t:Token, m: Tree) -> Tree:
#
#     for i in range(0,len(m.children)):
#         if isinstance(m.children[i], Token) and clean_tok(t) == clean_tok(m.children[i]):
#             m.children.pop(i)
#             m.children.insert(i,Token("ID","'"+t.value+"'"))
#
#     return m


def descend_left(t: Tree):
    if isinstance(t, Token):
        return t
    if isinstance(t.children[0], Token):
        return t.children[0]
    else:
        return descend_left(t.children[0])


def delete_descending_left(t: Tree):
    if isinstance(t.children[0], Token):
        t.children[0] = None
    else:
        delete_descending_left(t.children[0])


def visit(t: object, match, red):
    if isinstance(t, Token):
        return match(t)
    elif isinstance(t, Tree):
        return visit(t.children, match, red)
    elif isinstance(t, type([])):
        a = []
        for i in t:
            b = visit(i, match, red)
            a.append(b)
        return reduce(red, a)
    else:
        # cprint(type(t),"blue")
        raise Exception("Non esiste questo caso nella fun scrivi")


def apply(t: object, flist, ftok):
    if isinstance(t, Token):
        return ftok(t)
    elif isinstance(t, Tree):
        return Tree(t.data, apply(t.children, flist, ftok))
    elif isinstance(t, type([])):
        a = []
        for i in t:
            b = apply(i, flist, ftok)
            a.append(b)
        return flist(a)
    else:
        # cprint(type(t),"blue")
        raise Exception("Non esiste questo caso nella fun scrivi")


def sub_comp_sq_w_sq(t: list):
    if match(t, [0, 1, 2, 3, 4], ["ID", "EQ", "SQ", "W", "SQ"]) and len(t) == 5:
        m = [Token("ID", clean_tok(t[0]).value), Token("EQ", "="), Token("ID", "'" + clean_tok(t[3]).value + "'")]
        return m
    else:
        return t


def substitute_comp_SQ_W_SQ(t):
    t = apply(t, sub_comp_sq_w_sq, lambda x: x)
    return t


def escape(t: object):
    if isinstance(t, Token):
        return t

    if isinstance(t, list):

        if match(t, [0, 1, 2], ["SQ", "W", "SQ"]) and len(t) == 3:
            # print("before:", t)
            t = [Token("ID", "'" + t[1].value + "'")]
            # print("after:", t)
        if match(t, [0, 1, 2, 3, 4], ["ID", "EQ", "SQ", "W", "SQ"]) and len(t) == 5:
            # print("before:", t)
            t = [t[0], t[1], Token("ID", "'" + t[3].value + "'")]
            # print("after:", t)

        return t

    if isinstance(t, Tree):
        return Tree(t.data, escape(t.children))

    return t


class Toggler:

    def __init__(self):
        self.d = []

    def t(self, w: str):
        if w not in self.d:
            self.d.append(w)
            return True
        else:
            return False


def wellformed(t: object) -> bool:
    if isinstance(t, Token):
        if isinstance(t.value, str):
            return True
        else:
            return False

    if isinstance(t, list):
        # controllo se quel che c'è dentro è ben formato
        a = reduce(AND, map(wellformed, t))
        # controllo se io sono ben formato
        b = reduce(AND, map(lambda x: True if isinstance(x, Token) or isinstance(x, Tree) else False, t))
        return a and b

    if isinstance(t, Tree):
        a = True if isinstance(t.children, list) else False
        b = wellformed(t.children)
        return a and b

    return False


def create_macro_mod(mll, t):
    s = clean_tok(t.children[0])
    mll.macros[s] = apply(t.children[2],
                          lambda x: x,
                          lambda x: Token("ID", "'" + s + "'") if s == clean_tok(x).value else x)


def create_macro_exp(mll, t):
    s = clean_tok(t.children[0])
    mll.macros[s] = apply(t.children[2],
                          lambda x: x,
                          lambda x: Token("ID", "'" + s + "'") if s == clean_tok(x).value else x)


def create_macro_pip(mll, t):
    s = clean_tok(t.children[0]).value
    mll.macros[s] = Tree(t.data, apply(t.children[2:],
                                       lambda x: x,
                                       lambda x: Token("ID", "'" + s + "'") if s == clean_tok(x).value else x))


def cleanNone(t):
    if isinstance(t, Token):
        return t

    if isinstance(t, list):
        return filter(lambda x: x is not None, [cleanNone(x) for x in t])

    if isinstance(t, Tree):
        return t

    raise Exception("caso non previsto")


def SUM(a, b):
    return a + b


def leaves_before(t) -> int:
    if type(t) == Tree:
        if t.data=="macro":
            return 0
        return int(leaves_before(t.children))
    if type(t) == Token:
        if t.type == "CO" or t.type == "LP" or t.type == "RP":
            return 0.5
        return 1
    if isinstance(t, list):
        return reduce(SUM, map(leaves_before, t))

    raise Exception("Errorfffffffff")

def leaves_after(t) -> int:
    if type(t) == Tree:
        return int(leaves_after(t.children))
    if type(t) == Token:
        if t.type == "CO" or t.type == "LP" or t.type == "RP":
            return 0
        return 1
    if isinstance(t, list):
        return reduce(SUM, map(leaves_after, t))

    raise Exception("Errorfffffffff")


def give_type_agnostically(t) -> str:
    if isinstance(t,Tree):
        return str(t.data)
    if isinstance(t,Token):
        return str(t.type)


# def denest_sums(t:Tree) -> []:
#     print(list_types_list(t.children))
#
#     if isTree(t) and match(t.children,[],["PLUS"]) and len(t.children)==1:
#         return [t.children]
#
#     if isTree(t) and give_type_agnostically(t.children[0]).islower() and give_type_agnostically(t.children[2]).islower():
#         return denest_sums(t.children[0]) + denest_sums(t.children[2])
#     if isTree(t) and give_type_agnostically(t.children[0]).islower() and give_type_agnostically(t.children[2]).isupper():
#         return denest_sums(t.children[0]) + [[t.children[2]]]
#     if isTree(t) and give_type_agnostically(t.children[0]).isupper() and give_type_agnostically(t.children[2]).islower():
#         return [[t.children[0]]] + denest_sums(t.children[2])
#     if isTree(t) and give_type_agnostically(t.children[0]).isupper() and give_type_agnostically(t.children[2]).isupper():
#         return [[t.children[0]], [t.children[2]]]
#
#     return []