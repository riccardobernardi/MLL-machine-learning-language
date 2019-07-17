new_grammar = """

//////////////PARSER RULES

// | macro_exp | macro_mod | macro_pip

mll : ( model | macro | parmac | comment | summa )+

model : ID COLON [_rc] [PI] e (_nn)*

_rc.2 : RR | CC

_nn : ( PI e )

pyt : "!" (W | NUM | "," | " " | ":" | "+" | "=" | "[" | "]"| "(" | ")" | "'" )* WSS+ WSP*

comment : HASH (W | NUM | "," | " " | ":" | "+" | "=" | "[" | "]" | "(" | ")" | "'" )* WSS+ WSP*

parmac : ID DOLLAR ID (OR ID)*

summa : ID SCO ID

macro : ID EQC [ID] e

e   : ID
    | _mm
    | LP [e] RP // applicazione di funzione
    | NUMBER
    | SQ W SQ
    | e PLUS e
    | e MULT e
    | e SUB e
    | e DIV e
    | AT ID LP RP
    | AT e
    | ID (e | comp )+
    
_mm.2 : ( ID AR )
    | ( ID AR ID ID )
            
comp: ID EQ LSP (e 
            | e COLON
            | e CO )+ RSP 
    | ID EQ LP (e 
            | e CO)+ RP 
    | ID EQ BL (e 
            | e COLON)+ BR 
    | ID EQ SQ W SQ
    | ID EQ NUMBER
    | ID EQ ID
    | ID LP ( e CO )+ e RP 
    | LP ( e CO )+ e RP 
    
//////////////LEXER TERMINALS

SC : ";"

DOLLAR : "$"

SCO : "+:"

MP : "p:"

ME : "e:"

MM : "m:"

MULT : "*" WS

OR : "or" WS

AT : "@"

SUB : "-" WS

DIV : "/" WS

AR : "->" WS

EX : "!" WS

HASH : WS "#" [" "]+

RR : ("REGRESSOR" | "regressor") WS

CC : ("CLASSIFIER" | "classifier") WS

IS : "is" WS

EQC : ":=" WS

PLUS : "+" WS

WITH  : "with" WS

PI : "|" WS

CO : "," WS

DO  : "." WS

SQ : "'" WS

EQ : "=" WS

BR : "}" WS

BL  : "{" WS

LP : "(" WS

RP : ")" WS

LSP : "[" WS

RSP : "]" WS

ID : WS W [INTEGER | W | DO W]+ WS

NWID : W [INTEGER | W]+

WID : W [INTEGER | W]+

COLON   : ":" WS

W   : ("a".."z" | "A".."Z" | "_" )+

WS : (" " | "\\n" | "\\t" | "\\r")*

WSP : (" " | "\\n" | "\\t" | "\\r")+

WSS : ("\\n")

INTEGER  :   ("0".."9")+

DECIMAL  :   INTEGER ["." INTEGER]

NUMBER   :   NUM WS

NUM : (DECIMAL | INTEGER)

"""

def get_rev_grammar():
    return new_grammar
