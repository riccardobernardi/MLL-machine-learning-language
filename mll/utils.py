from keras import layers
from lark import Tree, Token

#stampa un albero a schermo, non scrive su stringhe
def stampa(t : object) -> None:
    if isinstance(t, Token):
        print(t.value,end='')
    elif isinstance(t, Tree):
        stampa(t.children)
    elif isinstance(t, type([])):
        for i in t:
            stampa(i)
    else:
        raise Exception("Non esiste questo caso nella fun stampa")


# ritorna una stringa con l' albero dentro
def scrivi(t : object) -> str:
    if isinstance(t, Token):
        return t
    elif isinstance(t, Tree):
        return scrivi(t.children)
    elif isinstance(t, type([])):
        s = ""
        for i in t:
            s += scrivi(i)
        return s
    else:
        raise Exception("Non esiste questo caso nella fun scrivi")


imports = """import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.model_selection import *
from mlxtend.classifier import StackingClassifier
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import InputLayer
from keras.layers import AveragePooling2D

def assign (x):
    return x

"""

def get_imports() -> str:
    return imports


def istok(i : object) -> Token :
    if isinstance(i,Token):
        return True
    else:
        return False


def clean_tok(value:str) -> str :
    return value.replace(" ","").replace("\n","")


def plus_in_array(t:list) -> bool:
    for i in t:
        if istok(i):
            if clean_tok(i) == "+":
                return True

    return False


def clean_tok_mod(tok: Token) -> Token:
    if tok.type != "P" and tok.type != "PP" and tok.type != "TAB" and tok.type != "DEF" and tok.type != "RETURN":
        return Token(tok.type,tok.value.replace(" ","").replace("\n",""))
    else:
        return tok


def clean_deep(t:list) -> list:
    if isinstance(t, Token):
        return clean_tok_mod(t)
    elif isinstance(t, Tree):
        return Tree(t.data,clean_deep(t.children))
    elif isinstance(t, list):
        return [clean_deep(i) for i in t]
    else:
        raise Exception("Non esiste questo caso nella fun clean_deep")


def clean_arr(t:list)->list:
    arr = []
    for i in t:
        if isinstance(i, Token) and i.type != "P" and len(clean_tok(i.value).replace("with", "")) == 0:
            pass
        else:
            arr += [i]
    t = arr
    return t


def escape(m:str, t:object) -> object:
    if isinstance(t, Token):
        return t
    elif isinstance(t, Tree):
        return Tree(t.data,escape(m,t.children))

    elif isinstance(t, type([])):
        i = 0
        while True:
            if i == len(t):
                break
            else:
                if istok(t[i]) and clean_tok(t[i].value) == m:
                    t[i].value = "'"+clean_tok(t[i].value)+"'"
                    if i+1< len(t) and istok(t[i+1]) and t[i+1].type == "SQ":
                        t.remove(t[i + 1])
                    if i-1>= 0 and istok(t[i-1]) and t[i-1].type == "SQ":
                        t.remove(t[i - 1])
                        i-=1
                i+=1
        return [escape(m,i) for i in t]

    else:
        raise Exception("Non esiste questo caso nella fun escape")


def get_keras_layers()-> set :
    arr = []
    keras_layers = set()

    for k in layers.__dict__.keys():
        if "__" not in k and k != "K":
            arr += [k]

    for i in arr:
        keras_layers.add(str(i))

    return keras_layers


def uncomma(s:str) -> str:
    if ",," in s:
        s=s.replace(",,",",")
        return uncomma(s)
    return s


def isTree(t : object) -> bool:
    if isinstance(t,Tree):
        return True
    return False

def CyclerTemplate(c:list)->list:

    i = 0
    while True:
        if i == len(c):
            break
        else:
            print(c[i])
            if i == 1:
                c.insert(2, 10)
            i += 1


def presentation():
    print()
    print("######################################################################")
    print("#                            STARTING MLL                            #")
    print("######################################################################")
    print("# 1. model = MLL('PROGRAM.MLL')                                      #")
    print("# 2. model.start() -> to pass from MLL to python                     #")
    print("# 3. model.get_string() -> to get python code of your program        #")
    print("# 4. model.execute() -> to run python code of your program           #")
    print("# 5. clf = model.last_model() -> to get last model of your program   #")
    print("# 6. MLL() -> to get this window                                     #")
    print("#                                                                    #")
    print("#                                    student: Bernardi Riccardo      #")
    print("#                                    supervisor: Lucchese Claudio    #")
    print("#                                    co-supervisor: Spanò Alvise     #")
    print("######################################################################")

def list_types(t:list) -> None:
    for n,i in enumerate(t):
        if isinstance(i,Token):
            print(str(n) + " " + str(type(i))+ " " + clean_tok(i.value))
        else:
            print(str(n) + " " + str(type(i)))


def clean_tabs(t:list) -> list:
    return Token("COMMENT",scrivi(t).replace("\t","").replace("\n","")+"\n")