from lark import Transformer
from .graphdef import QueryGraphDef

class QueryTransformer(Transformer):
    true_val = lambda self, _: True
    false_val = lambda self, _: False
    def string(self, str):
        s = str[0]
        if s[0] == '"':
            return str[0][1:-1]
        else:
            return s
    def number_int(self, n):
        return int(n[0])

    def number_float(self, n):
        return float(n[0])
    def args(self, items):
        if len(items) == 1:
            return items[0]
        else:
            return items

    def pair(self, items):
        return {items[0]: items[1]}

    def map(self, items):
        return items[0]


    def id(self, items):
        return str(items[0])

    def process_args(self, items):
        if type(items) != list:
            items = [items]
        args=[]
        kwargs={}
        for arg in items:
            if type(arg) == dict:
                kwargs.update(arg)
            else:
                args.append(arg)
        return (args, kwargs)

    def node(self, items):

        if len(items) == 1:
            return (items[0], None)
        else:
            # process args
            (args, kwargs) = self.process_args(items[1])

            return (items[0], (args, kwargs))

    def edge(self, items):
        if len(items) == 0:
            return (None, None)
        else:
            # process args
            (args, kwargs) = self.process_args(items[0])
            return (None, (args, kwargs))

    def empty_graph(self, items):
        return QueryGraphDef()

    def node_graph(self, items):
        _id = set([items[0][0]])
        return QueryGraphDef(_id, _id, steps=[items[0]])

    def edge_graph(self, items):
        g1 = items[0]
        g2 = items[2]

        new_steps = []
        edge_arg = items[1][1]
        for s in g1.snk:
            for t in g2.src:
                new_steps.append((edge_arg, s, t))

        return QueryGraphDef(src=g1.src, snk=g2.snk, steps=g1.steps + g2.steps + new_steps)

    @staticmethod
    def merge_graphs(items):
        gdef = QueryGraphDef()
        for g in items:
            gdef.src.update(g.src)
            gdef.snk.update(g.snk)
            gdef.steps = gdef.steps + g.steps
        return gdef

    def group_graph(self, items):
        return QueryTransformer.merge_graphs(items)
