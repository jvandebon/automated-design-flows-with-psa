from lark import Lark
from .transformer import QueryTransformer

query_grammar = r"""

id              : /[a-zA-Z][a-zA-Z0-9_]*/

pair: id ":" value

value: pair -> map
    | ESCAPED_STRING -> string
    | SIGNED_INT -> number_int
    | SIGNED_FLOAT -> number_float
    | "True" -> true_val
    | "False" -> false_val
    | id -> string

args            : value
                | args ("," value)*

node            :   id
                  | id "{" args "}"

edge            : "=>"
                | "={" args "}>"

graph           : "(" graph ("|" graph)* ")" -> group_graph
                | "0" -> empty_graph
                | node -> node_graph
                | graph edge graph -> edge_graph



%import common.ESCAPED_STRING
%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.WS
%ignore WS
"""
from lark import Lark

class QueryEngine:
    def __init__(self):
        global query_grammar
        self.grammar = Lark(query_grammar, start='graph', parser='lalr')
        self.transformer = QueryTransformer()

    def process(self, qgraph, vx_args, eg_args):
        query = qgraph.select.replace('\n', '')
        queries = []
        for q in query.split(";"):
            _q = q.strip()
            if _q != '':
                queries.append(_q)

        graph_defs = []
        for q in queries:
           pre = self.grammar.parse(q)
           graph_def = self.transformer.transform(pre)
           graph_defs.append(graph_def)

        if len(graph_defs) == 1:
            graph_def = graph_defs[0]
        else:
            graph_def = QueryTransformer.merge_graphs(graph_defs)

        qgraph = graph_def.build(qgraph, vx_args, eg_args)

        return qgraph