new_grammar = """

//////////////PARSER RULES

mll : ( model | macro | parmac | dag | comment )*

model : [mt] ID COLON ID [e*] [WITH comp*] (PLUS (ID [e*] [WITH comp*]) )*
        
macro : ID EQC (_msid | _mse | dotname)

c2d: C2D FILTER KERNEL SUBSAMPLE BORDER [ACTIVATION]

m2d: M2D KERNEL STRIDES BORDER [ACTIVATION]
        
dotname.3 : ID DO ID (DO ID)* 

_msid.2 : ID [e*] [WITH comp*] (PLUS (ID [e*] [WITH comp*]) )*

_mse : [e*] [WITH comp*] (PLUS (ID [e*] [WITH comp*]) )*

dag : ID COLON ( PI (ID AR | ID) [e*] [WITH comp*] [(PLUS (ID [e*] [WITH comp*]) )*] )+

//[COLON ID [e*] [WITH comp*] (PLUS (ID [e*] [WITH comp*]) )*]

mt.2 : RR | CC //model type

comment : HASH (FF | CO | " ")* WSP

//macro per i parametri
parmac.2 : ID IS n (OR n)*

comp : ID EQ e
        | ID EQ LSP e [CO] (CO e)* RSP
        | ID EQ ID DO ID LSP e [COLON] (COLON e)* RSP

e   : BL n COLON e (CO n COLON e )*  BR
    | ID
    | LP e* RP
    | NUMBER
    | n
    | e CO
    | fapp
    
    
fapp.2 : FEXTNAME LP e* RP

//tolto perchè sennò veniva piallato il @ext con le parentesi successive come applicazione di metodo
//invece che come variabile e tupla    
//| ID LP (e)* RP    
    
//    | l

// j.2 : WITH e+

n   : SQ W SQ

//l   : e | e PI l

//////////////LEXER TERMINALS

FEXTNAME : ["@"] FF

FILTER : (NUM)+

KERNEL : NUM NUM

SUBSAMPLE : NUM NUM

STRIDES : NUM NUM

BORDER : "v" | "s"

ACTIVATION : "l" | "r" //linear or relu

C2D : "c2d"

M2D : "m2d"

AR : "->" WS

GO : "|>" WS

EX : "!" WS

DAG : "DAG" WS

HASH : WS "#" " "*

RR : ("REGRESSOR" | "regressor") WS

CC : ("CLASSIFIER" | "classifier") WS

IS : ("IS" | "is") WS

OR : "or" WS

EQC : ":=" WS

PLUS : "+" WS

WITH  : "with" WS

PI : "|" WS

CO : WS "," WS

DO  : "." WS

SQ : "'" WS

EQ : "=" WS

BR : "}" WS

BL  : "{" WS

LP : "(" WS

RP : ")" WS

LSP : "["

RSP : "]"

ID : (XID | IID) WS

XID : "@" W ( INTEGER | W )* WS

IID : WS W ( INTEGER | W )* WS

FF : W ( INTEGER | W )*

COLON   : ":" WS

W   : ("a".."z" | "A".."Z" | "_" )+

WS : (" " | "\\n" | "\\t" | "\\r")*

WSP : (" " | "\\n" | "\\t" | "\\r")+

INTEGER  :   ("0".."9")+

DECIMAL  :   INTEGER ("." INTEGER)?

NUMBER   :   NUM WS

NUM : (DECIMAL | INTEGER)

"""

def get_new_grammar():
    return new_grammar



#se invece di avere macro sfruttassi a palla il dict?
# criterion IS 'gini' 'entropy'
#foreach val after IS:
#   create dict entry with "before IS" + "=" + "val"

#l' assegnamento di layers ad un layer non necessita di un nuovo tipo di assegnamento!!!
