from ..hgraph import HGraph
from .engine import QueryEngine


class QGraph(HGraph):
    def __init__(self, select, vx_args, eg_args):
        def ginit(graph):
            graph.pmap['ids'] = { }
            graph.vstyle['label'] = lambda g, vx: r'''<<TABLE CELLBORDER="0" CELLSPACING="0" border="0"><TR align="right"><TD><B>%s:</B>%d %s</TD></TR></TABLE>>''' % (g.pmap[vx]['id'], vx,  g.pmap[vx]['args'] )
            graph.estyle['label'] = lambda g, eg: "%s" % (g.pmap[eg]['args'])
            graph.vstyle['shape'] = "component"
            graph.vstyle['fillcolor'] = "burlywood1"

        super().__init__(ginit=ginit)
        self.select = select

        engine = QueryEngine()
        engine.process(self, vx_args=vx_args, eg_args=eg_args)

