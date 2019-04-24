# def model_operations(self,t:list) -> list:
#     '''
#     se ho un model voglio che dato un list in input in questa funzione in putput mi esca lo stesso list
#     che però dovrà avere parentesi alla fine e valutando se ci sono delle + all interno dovrò capire se
#     mettere delle add e come mettere le parentesi agli identificatori di metodi
#     :param t:
#     :return:
#     '''
#
#     plus_in_model=False
#
#     for i in t:
#         if istok(i):
#             i.value = i.value.replace(" ","").replace("\n","")
#             if "+" in i.value:
#                 plus_in_model=True
#
#     for i in t:
#         if istok(i):
#             i.value = i.value.replace(" ", "").replace("\n", "")
#             if i.value=="Sequential":
#                 print("---------------"+self.ordered_models[len(self.ordered_models)-1].__str__())
#                 self.current_keras = self.ordered_models[len(self.ordered_models)-1]
#
#     self.ordered_models.append(t[0].value.replace(" ","").replace("\n",""))
#     self.models[t[0].value.replace(" ","").replace("\n","")] = 0
#
#     t[0].value = ("models[' " + t[0].value.replace(" ","").replace("\n","") + " ']")
#
#     t[1].value = "="
#
#     if plus_in_model:
#         t.append(Token("P", "))"))
#     else:
#         t.append( Token("P", ")\n"))
#
#     t.insert(3,Token("P","("))
#
#     return t


# def comp_operations(self,t):
#     pass
#     # for i in t:
#     #     if istok(i):
#     #         i.value = i.value.replace(" ","").replace("\n","")


# for i in range(0, len(t)):
#     if istok(t[i]) and clean_tok(t[i].value)== "+":
#         t.insert(i-1,Token("P","))\n"))
#         t[i].value = self.current_keras + ".add("


# t.insert(index,self.macros[sub])
                #print("parola corrente : " + t[i].value.__str__())


# t.remove(i)


# for i in t:
#     if istok(i) and clean_tok(i.value)== "+":
#         i.value = self.current_keras + ".add("


# for i in range(0, len(t)):
#     if istok(t[i]) and clean_tok(t[i].value)== "+":
#         t.insert(i-1,Token("P","))\n"))
#         t[i].value = self.current_keras + ".add("

# print("format-keras"+str(type(t)))


#
# if l != "P":
#     s = s.replace("\n","")
#
# # if "\n" in s:
# #     if(s.replace("\n","") in self.macros.keys()):
# #         s= str(self.macros[s.replace("\n","")])+"\n"
# # else:
# #     if (s.replace("\n", "") in self.macros.keys()):
# #         s = str(self.macros[s.replace("\n", "")])
#


#
# # cose per le funzioni
#
# if l == "ID" and "\n" in s and "@" in s and "with" not in s:
#     #s = s.replace("\n", "()\n", 1)
#     s = s.replace("@", "")
# else:
#
#     # if "+" in s:
#     #     s= ").add("
#
#     # regola indipendente: nessun altro tocca il "\n" prima di lui
#     if "\n" in s:
#         pass
#         #s = s.replace("\n", ")\n", 1)
#
#     # regola indipendente: nessun altro tocca il AT prima di lui


#
#     # regola indipendente: nessun altro tocca il COLON prima di lui
#     # if l == "COLON":
#     #     s = s.replace(":", "=")
#     if l=="EQC":
#         s = s.replace(":=", "=")
#
#     if "with," in s:
#         s = s.replace("with,", "")
#
#     # regola indipendente: nessun altro tocca il WITH e LPAREN prima di lui
#     if "with" in s:
#         s = s.replace("with", "")

# if len(s)==0:
#     print("len 0")


# for i in t.children:
#     for j in self.macros.keys():
#         if istok(i) and i.replace("\n","").replace(" ","") == j:
#             t.children = self.put_macros(t.children)

# for i in t.children:
#     for j in self.macros.keys():
#         if istok(i) and i.replace("\n","").replace(" ","") == j:
#             t.children = self.put_macros(t.children)


# print("----["+t.children.__str__()+"]")
# else:
#     #t.children = self.format_sklearn(t.children)
#     pass


# if self.plus_in_array(t.children):
#     t.children = self.format_keras(t.children)


# self.insert_parens(t)

# t = self.format_parenthesis(t)

# for j in self.macros.keys():
#     if istok(t) and t.replace("\n", "").replace(" ", "") == j:
#         t = self.put_macros([t])[0]

# print(self.macros[m])
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

# print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
# print("macro : "+t.__str__()+"\n\n")

# t = self.format_parenthesis(t)
# t = self.format_commas(t)

# stampa(t)
# assert(1==2)

# for j in range(0, round(len(t)/2)):
#     length = len(t)
#     for i in range(0, length):
#         # se trovo un + allora la funzione successiva guadagna una parentesi aperta e il +
#         # diventa una add
#
#         if istok(t[i]) and clean_tok(t[i].value)== "+" and opened:
#             t.insert(i, Token("P", "))\n"))
#             opened = False
#             i += 1
#
#         if istok(t[i]) and clean_tok(t[i].value)== "+":
#
#             t[i].value = self.current_keras
#             t.insert(i+1,Token("ADD",".add("))
#             i+=1
#
#             opened = True
#             if i + 2 < len(t):
#                 t.insert(i + 2, Token("P", "("))
#
# t.append(Token("PP","))\n"))

# grammar = """
#
# //////////////PARSER RULES
#
# mll : model+
#
# model : IID COLON ID e*
#
# e   : j
#     | BR (n EQ e) (CO (n EQ e) )*   BL
#     | ID
#     | ID LP (e)* RP
#     | LP e* RP
#     | ID EQ e
#     | m
#     | NUMBER
#     | n
#     | e CO
# //    | l
#
# j.2 : WITH e+
#
# m   : LSP e+ RSP
#
# n   : SQ W SQ
#
# //l   : e | e PI l
#
# //////////////LEXER TERMINALS
#
# WITH  : "with" WS
#
# PI : "|" WS
#
# CO : WS "," WS
#
# DO  : "." WS
#
# SQ : "'" WS
#
# EQ : "=" WS
#
# BR : "}" WS
#
# BL  : "{" WS
#
# LP : "(" WS
#
# RP : ")" WS
#
# LSP : "["
#
# RSP : "]"
#
# ID : XID | IID WS
#
# XID : "@" W WS
#
# IID : WS W WS
#
# COLON   : ":" WS
#
# W : ("a".."z" | "A".."Z" | "_")+ WS
#
# WS : (" " | "\\n" | "\\t" | "\\r")*
#
# INTEGER  :   ("0".."9")+ WS
#
# DECIMAL  :   INTEGER ("." INTEGER)? WS
#
# NUMBER   :   DECIMAL | INTEGER WS
#
# """
#
# #use newer version
# def get_grammar():
#     return grammar

# if concat_in_array and istok(t[i]) and (clean_tok(t[i].value) == "|") and ((i + 1) < len(t)) and (
#         clean_tok(t[i + 1].value) == "concat"):
#     # allora devo mettere la concat degli ultimi modelli e poi devo subito svuotare quell array
#     t.remove(t[i])
#     t.remove(t[i])
#     # t.insert(i, Token("MODEL", "))(" + models[0] + ")"))
#     nn.append(Token("TAB", "\n\t"))
#     nn.append(Token("CONCAT", "x = concatenate(["))
#
#     s = []
#
#     for j in to_concat_and_free:
#         if to_concat_and_free.index(j) < (len(to_concat_and_free) - 1):
#             s += [j]
#             s += [","]
#         else:
#             s += [j]
#
#     lmm = ""
#     for j in s:
#         lmm += j
#
#     models.insert(0, names.pop(0))
#     nn.append(Token("MODELS", lmm))
#
#     # libero per i prossimi modelli che verranno concatenati
#     to_concat_and_free = []
#     write_concat = True




