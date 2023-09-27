import inspect
from heterograph.query.resultset import QueryResultSet
from heterograph.query.visitor import QueryVisitor
import html
from functools import partial
from tabulate import tabulate
from colorama import Fore, Style

class AstQueryResult(QueryResultSet):
    @staticmethod
    def label_vx(rset, ast, vx):
        cnode = ast[vx]
        header = cnode.entity
        range = cnode.range

        loc = html.escape(f"({range})")

        if vx in rset:
            matches=r"<TR><TD><B><I><FONT COLOR='#6600ff'>%s</FONT></I></B></TD></TR>" % str(rset[vx])[1:-1].replace("'", "")
        else:
            matches=""

        spec = r'<TABLE BORDER="0"><TR><TD ALIGN="CENTER"><B>%s (%d)</B></TD></TR><TR><TD ALIGN="CENTER"><b><FONT COLOR="#BA2E00"> %s</FONT></b></TD></TR>%s</TABLE>' % (header, vx, loc, matches)
        return r'<%s>' % spec

    def view(self, host='0.0.0.0', port='8888', viewer=None, **kwargs):
        ast = self.g

        if viewer is None:
            from ..core.webview import WebView
            _viewer = WebView()
        else:
            _viewer = viewer

        _viewer.add_graph(graph=self.qgraph, title='query pattern')

        # rset: { vx_a: set{ id0, id1, ... } }
        rset = { }

        for r in self.matches:
            match = zip(self.ids, r)
            for m in match:
                vx = m[1]
                if vx not in rset:
                    rset[vx] = set()
                rset[vx].add(m[0])

        vstyle = {}
        vstyle['fillcolor'] = lambda g, v: "burlywood1" if v in rset else "gray90"
        vstyle['label'] = partial(AstQueryResult.label_vx, rset)

        nmatches = len(self.matches)

        _viewer.add_graph(graph=self.g, title='query result: %s, %d match%s' % (self.qgraph.select, nmatches, "es" if nmatches != 1 else ""), vstyle=vstyle)

        if viewer is None:
            _viewer.run(host=host, port=port, **kwargs)

    def __repr__(self):

        g_vs = set(self.g.vertices)

        def fmt(match):
            fmatch = []

            for vx in match:
                if vx is None:
                    fm = f"{Fore.LIGHTRED_EX}---{Style.RESET_ALL}"

                elif vx not in g_vs:
                    fm = f"{Fore.LIGHTRED_EX}vertex '{vx}' deleted! {Style.RESET_ALL}"
                else:
                    cnode = self.g[vx]
                    if not cnode.isentity('Module'):
                        unparse = cnode.unparse()
                        wrap = 600
                        if len(unparse) > wrap:
                            unparse = unparse[0:wrap] + "..."
                    else:
                        unparse = ""

                    if len(unparse) == 0:
                        fm = f"{Fore.LIGHTRED_EX}{cnode.id}:{Fore.LIGHTBLUE_EX}{cnode.entity}{Fore.LIGHTBLACK_EX}({cnode.location}){Style.RESET_ALL}"
                    else:
                        fm = f"{Fore.LIGHTRED_EX}{cnode.id}:{Fore.LIGHTBLUE_EX}{cnode.entity}{Fore.LIGHTBLACK_EX}({cnode.location}):{Style.RESET_ALL}{unparse}"
                fmatch.append(fm)
            return fmatch

        data = [ fmt(match) for match in self.matches ]
        out = tabulate(data, headers=self.ids, tablefmt="fancy_grid")
        return out


def query_visitor(g, vs, select, where=None, **kwargs):
    def vx_args(qgraph, vx, entity=None):
        if entity is None:
            entity = 'node'
        qgraph.pmap[vx]['entity'] = entity

    def eg_args(qgraph, eg, *args, **kwargs):
        if args and kwargs:
            raise RuntimeError("cannot mix positional arguments with keyword arguments when defining edge attributes!")
        if args:
            if len(args) == 1:
                min = max = args[0]
            elif len(args) == 2:
                min = args[0]
                max = args[1]
        elif kwargs:
            min = kwargs.get('min', 0)
            max = kwargs.get('max', 0)
        else:
            min = 0
            max = 0

        qgraph.pmap[eg]['min_dist'] = min
        qgraph.pmap[eg]['max_dist'] = max

    def path_check(g, qgraph, edge, qedge):
        # eg => (pvx, vx)
        # qeg => (qpvx, qvx)

        (prefix_vx, vx) = edge
        (prefix_qvx, qvx) = qedge
        qgraph_entity = qgraph.pmap[qvx]['entity']
        match_entity = g[vx].isentity(qgraph_entity)

        if match_entity:

            # check distance constaints
            if (prefix_vx is not None) and (prefix_qvx is not None):
                distance = g[vx].depth - g[prefix_vx].depth
                min_dist = qgraph.pmap[qedge]['min_dist']
                max_dist = qgraph.pmap[qedge]['max_dist']

                if min_dist != 0 and distance < min_dist:
                    return False

                if max_dist != 0 and distance > max_dist:
                    return False

        return match_entity

    def match_filter(g, qgraph, match):

        if where is None:
            return True

        args = { }
        for p in where_params:
            if p in match:
                args[p] = g[match[p]]
            else:
                args[p] = None
        return where(**args)

    qv = QueryVisitor()
    if where is not None:
        where_params = list(inspect.signature(where).parameters)

    (qgraph, results) = qv.run(g=g, select=select, vs=vs, vx_args=vx_args, eg_args=eg_args, path_check=path_check, match_filter=match_filter, **kwargs)

    return AstQueryResult(g, qgraph, results)