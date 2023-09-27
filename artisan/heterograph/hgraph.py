import graph_tool as gt
from graphviz import Digraph
import copy
import sys
from .algorithm.dfs import dfs_visit
import wrapt

@wrapt.decorator
def modifies_graph(wrapped, instance, args, kwargs):
    if instance.read_only:
        raise RuntimeError("cannot modify read-only graph!")
    return wrapped(*args, **kwargs)

class HGraphProps(dict):
    def __init__(self, g):
        self.g = g
        self.vx = { }
        self.eg = { }
        super().__init__()

    def copy_prop_elem(self, new_g, new_elem, elem):
        if type(elem) == int:
            self.g.check_vx(elem, verify=True)
            new_g.check_vx(new_elem, verify=True)
            # we only copy if it exists
            if elem in self.vx:
                new_g.pmap[new_elem] = copy.deepcopy(self.vx[elem])
        elif type(elem) == tuple and len(elem) == 2:
            self.g.check_edge(elem, verify=True)
            new_g.check_edge(new_elem, verify=True)
            if elem in self.eg:
                new_g.pmap[new_elem] = copy.deepcopy(self.eg[elem])
        else:
            raise RuntimeError("invalid element '%s' specified!" % str(key))


    def rm_elem(self, elem):
        if type(elem) == int:
            if elem in self.vx:
               del self.vx[elem]
        elif type(elem) == tuple and len(elem) == 2:
            if elem in self.eg:
               del self.eg[elem]
        else:
            raise RuntimeError("invalid element '%s' specified!" % str(key))

    def __getitem__(self, key):
        if type(key) == int:
            # vertex
            self.g.check_vx(key, verify=True)

            # dynamic generation of property map
            if key not in self.vx:
                self.vx[key] = { }
            return self.vx[key]

        if type(key) == tuple and len(key) == 2:
            self.g.check_edge(key, verify=True)
            if key not in self.eg:
                self.eg[key] = {}

            return self.eg[key]

        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if type(key) == int:
            # vertex
            if type(value) != dict:
               raise RuntimeError("property map value must be 'dict' type!")

            self.vx[key] = value
        elif type(key) == tuple and len(key) == 2:
            # vertex
            if type(value) != dict:
               raise RuntimeError("property map value must be 'dict' type!")

            self.eg[key] = value
        else:
           super().__setitem__(key, value)

class HGraph:
    def __reset(self):
        self.__g.clear()
        self.__vx_counter = 0 # vx counter

        # vx: used to identify vertex 
        self.__ivx = { } # vx => ivx

        # ivx: internal vx - graph_tool
        self.__vx = { } # ivx => vx

        # vx => [vx_in0, vx_in1] : inputs of vx
        self.__in = { }

        # vx => [vx_out0, vx_out1] : outputs of vx
        self.__out = { }

        # graph map properties
        self.__properties = HGraphProps(self)

        '''
        # hovering support
        self.on_hover = None
        # def hover(g, elem):
        #    if type(elem) == int:
        #       return "<HTML CODE>"
        '''

        # read-only support
        self.read_only = False

        # default style
        self.__gstyle = { 'layout': 'dot', 'rankdir': 'TD' }
        self.__vstyle = { 'shape': 'Mrecord', 'style': 'filled', 'fillcolor': '#99CCFF', 'label': lambda g, id: str(id) }
        self.__estyle = { 'color': '#777777', 'arrowhead': 'open' }

        # graph initialisation, including setting style
        if self.__ginit:
            self.__ginit(self)

    @property
    def igraph(self):
        return self.__g

    @property
    def to_ivx(self):
        return self.__ivx

    @property
    def to_vx(self):
        return self.__vx

    def __init__(self, *, ginit=None, vinit=None, einit=None):
        # graph tool
        self.__g = gt.Graph(g=None, directed=True, prune=False, vorder=None)
        self.__g.set_fast_edge_removal(fast=True)

        self.__ginit = ginit
        self.__vinit = vinit
        self.__einit = einit

        self.__reset()


    #################################### style

    # g.[v|e]style = None empties style, otherwise it adds/updates
    @property
    def style(self):
        return self.__gstyle

    @style.setter
    def style(self, gstyle):
        if gstyle is None:
            self.__gstyle = { }
        else:
            if type(gstyle) != dict:
               raise RuntimeError("style must be a dict type")
            self.__gstyle.update(gstyle)

    @property
    def vstyle(self):
        return self.__vstyle

    @vstyle.setter
    def vstyle(self, vstyle):
        if vstyle is None:
            self.__vstyle = { }
        else:
            if type(vstyle) != dict:
               raise RuntimeError("style must be a dict type")
            self.__vstyle.update(vstyle)

    @property
    def estyle(self):
        return self.__estyle

    @estyle.setter
    def estyle(self, estyle):
        if estyle is None:
            self.__estyle = { }
        else:
            if type(estyle) != dict:
               raise RuntimeError("style must be a dict type")
            self.__estyle.update(estyle)

    #################################### property maps

    # g.pmap <= returns graph properties
    # g.pmap[vx] <= returns vertex properties
    # g.pmap[edge] <= returns edge properties
    #

    @property
    def pmap(self):
       return self.__properties

    @pmap.setter
    def pmap(self, value):
        self.__properties.clear()
        if type(value) != dict:
            raise RuntimeError("property map must be set using a dictionary!")
        self.__properties.update(value)

    def __gen_vx_id(self):
        _id = self.__vx_counter
        self.__vx_counter = self.__vx_counter + 1
        return _id

    def __to_ivs(self, vs):
        ret = {}
        for vx in vs:
            ivx = self.__ivx.get(vx, None)
            if ivx is None:
                raise RuntimeError("cannot find internal vertex for id: %s" % vx)
            ret[ivx] = vx
        return ret

    @modifies_graph
    def erase(self):
        self.__reset()

    def copy(self, *, vs=None, g=None, induced=True, ret_map=False):
        if g is None:
            g = HGraph(ginit=self.__ginit, vinit=self.__vinit, einit=self.__einit)
            g.style = None; g.style = self.style
            g.vstyle = None; g.vstyle = self.vstyle
            g.estyle = None; g.estyle = self.estyle

        g.read_only = False

        # copy graph properties
        g.pmap = dict(self.__properties)


        if vs is None:
            vs = self.vertices        

        _map = { } # vx(self) => vx (g)
        for vx in vs:
            _vx = g.add_vx(1)
            _map[vx] = _vx
            self.__properties.copy_prop_elem(g, _vx, vx)

        if induced:
            for e in self.edges:
                _e0 = _map.get(e[0], None)
                _e1 = _map.get(e[1], None)
                if (_e0 is not None) and (_e1 is not None):
                    g.add_edge(_e0, _e1)
                    self.__properties.copy_prop_elem(g, (_e0, _e1), e)

        g.read_only = self.read_only

        if ret_map:
            return (g, _map)
        else:
            return g

    @modifies_graph
    def remove_subgraph(self, vx):
        # vx is the root of the subgraph we wish to remove
        def visit(g, vx, synth):
           synth.append(vx)
           return synth

        vs = dfs_visit(g=self, vx=vx, post=visit)
        self.rm_vx(vs)


    ################################# vertex
    @property
    def num_vx(self):
        return self.__g.num_vertices()

    @property
    def vertices(self):
        return [*self.__ivx]

    @property
    def source(self):
        return [v for v in self.vertices if len(self.in_vx(v)) == 0]

    @property
    def sink(self):
        return [v for v in self.vertices if len(self.out_vx(v)) == 0]

    def __neighbours(self, _in, vx, order=None, after=True, anchor=None):
        # check if vx exists
        self.check_vx(vx)

        if _in:
            nb=self.__in.get(vx, [])
        else:
            nb=self.__out.get(vx, [])

        if (order is not None) and (len(order) > 0):
            if type(order) == int:
                order = [order]

            if not set(order).issubset(set(nb)):
                raise RuntimeError("specified order %s is not a subset of neighbours of vertex %d: %s!" % (order, vx, nb))

            # remove elements from list
            for x in order:
                nb.remove(x)

            if anchor is None:
                if after:
                    nb.extend(order)
                else:
                    nb = order + nb
            elif type(anchor) == int:
                try:
                   pos = nb.index(anchor)
                except ValueError:
                    raise RuntimeError("anchor '%d' not found!" % anchor)

                if after:
                    nb = nb[0:pos+1] + order + nb[pos+1:]
                else:
                    nb = nb[0:pos] + order + nb[pos:]
            else:
                raise RuntimeError("invalid anchor '%s': must be an int!" % anchor)

        return nb

    def num_in_vx(self, vx):
        ret = self.__in.get(vx, [])
        return len(ret)

    def num_out_vx(self, vx):
        ret = self.__out.get(vx, [])
        return len(ret)

    def out_vx(self, vx, *, order=None, after=True, anchor=None):
       if order is not None and self.read_only:
           raise RuntimeError("cannot modify read-only graph!")
       return self.__neighbours(_in=False, vx=vx, order=order, after=after, anchor=anchor)


    def in_vx(self, vx, *, order=None, after=True, anchor=None):
       if order is not None and self.read_only:
           raise RuntimeError("cannot modify read-only graph!")
       return self.__neighbours(_in=True, vx=vx, order=order, after=after, anchor=anchor)

    @modifies_graph
    def add_vx(self, n=1):
        ivs = self.__g.add_vertex(n)

        if n == 1:
            _ivs = [ivs]
        else:
            _ivs = ivs

        ret = []
        for _ivx in _ivs:
            ivx = int(_ivx)
            vx = self.__gen_vx_id()

            self.__ivx[vx] = ivx
            self.__vx[ivx] = vx

            ret.append(vx)

            if self.__vinit:
                self.__vinit(self, vx)

        return ret if n > 1 else ret[0]

    @modifies_graph
    def rm_vx(self, vs):
        if type(vs) == int:
            vs = [vs]

        if len(vs) == 0:
            raise RuntimeError("no vertex ID specified!")


        ivs = self.__to_ivs(vs) # {ivx: vx}

        # reverse order
        for ivx in reversed(sorted(ivs)):
            self.__g.remove_vertex(ivx)
            vx = ivs[ivx]
            del self.__ivx[vx]

        # defragment
        n = self.num_vx
        m = n
        for i in reversed(range(0, self.__vx_counter)):
            if i in self.__ivx:
                n = n - 1
                self.__ivx[i] = n
                self.__vx[n] = i
            if i >= m:
                self.__vx.pop(i, None)

        # bookkeeping
        for vx in vs:
            _in = self.__in.get(vx, None)
            if _in:
                for v in _in:
                    self.__out[v].remove(vx)

            _out = self.__out.get(vx, None)
            if _out:
                for v in _out:
                    self.__in[v].remove(vx)
            self.__in.pop(vx, None)
            self.__out.pop(vx, None)

            self.__properties.rm_elem(vx)


    def check_vx(self, vs, verify=False):
        if type(vs) == int:
            vs = [vs]

        for _id in vs:
            if _id not in self.__ivx:
                if verify:
                    raise RuntimeError("vertex '%d' is invalid!" % _id)
                else:
                    return False
        return True

    ################################# edges
    @property
    def num_edges(self):
        return self.__g.num_edges()

    @property
    def edges(self):
        return [ (self.__vx[int(e[0])], self.__vx[int(e[1])]) for e in self.__g.get_edges() ]

    @modifies_graph
    def add_edge(self, s, t):
        if type(s) == int:
            s = [s]
        if type(t) == int:
            t = [t]

        edges = []
        for _s in s:
            for _t in t:
                # make sure s and t exist
                ivx_edge = list(self.__to_ivs([_s, _t]).keys())

                # ignore self-cycles
                if _s == _t:
                    continue
                # ignore existing edges
                if self.__g.edge(ivx_edge[0], ivx_edge[1], add_missing=False) is not None:
                    continue
                self.__g.add_edge(ivx_edge[0], ivx_edge[1], add_missing=False)

                vx_edge = (_s, _t)
                edges.append(vx_edge)

                if self.__einit:
                    self.__einit(self, vx_edge)

                # bookkeeping
                if _t not in self.__in:
                    self.__in[_t] = [_s]
                else:
                    self.__in[_t].append(_s)

                if _s not in self.__out:
                    self.__out[_s] = [_t]
                else:
                    self.__out[_s].append(_t)

        return edges

    def check_edge(self, edge, verify=False):
        if type(edge) == tuple:
            edges = [edge]
        else:
            edges = edge

        for e in edges:

            found = True

            try:
                [ev0, ev1] = self.__to_ivs([e[0], e[1]])
            except RuntimeError:
                found = False

            if not found:
                if verify:
                    raise RuntimeError("edge %s is invalid!" % str(e))
                else:
                    return False

            if self.__g.edge(ev0, ev1, add_missing=False) is None:
                if verify:
                    raise RuntimeError("edge %s not found!" % str(e))
                else:
                    return False

        return True

    @modifies_graph
    def rm_edge(self, edge, verify=False):
        if type(edge) == tuple:
            if len(edge) != 2:
                raise RuntimeError("invalid edge: %s!" % str(edge))
            edges = [edge]
        else:
            edges = edge # list of edges

        g = self.__g
        for e in edges:
            try:
                [ev0, ev1] = self.__to_ivs([e[0], e[1]])
            except RuntimeError as ex:
                raise RuntimeError("invalid edge descriptor: %s" % str(e)) from ex

            edge = g.edge(ev0, ev1, add_missing=False)
            if edge:
               g.remove_edge(edge)
               # bookkeeping
               self.__in[e[1]].remove(e[0])
               self.__out[e[0]].remove(e[1])

               self.__properties.rm_elem(e)

            else:
                if verify:
                    raise RuntimeError("edge (%d, %d) does not exist, and thus cannot be removed!" % (e[0],e[1]))


    def render(self, *, filename='graph.svg', format='svg', pipe=False, vs=None, induced=True, gstyle=None, vstyle=None, estyle=None, **kwargs):
        # ==== cluster support ===
        # g.style['nclusters'] = 4
        # g.style['cluster'] = lambda g, c: {'label': 'abc'}
        # g.vstyle['cluster'] = lambda g, v: return c

        def init_styles(int_style, arg_style):
            style_n = { }; style_w = { }; style_c = { }
            _style = { }
            _style.update(int_style)
            if arg_style is not None:
               _style.update(arg_style)

            for s in _style:
                if s[0] == '#':
                    style_w[s[1:]] = _style[s]
                elif s in ['nclusters', 'cluster']:
                    style_c[s] = _style[s]
                elif s[0] != '!':
                    style_n[s] = _style[s]

            return (style_n, style_w, style_c)

        (_gstyle_n, _gstyle_w, _gstyle_c) = init_styles(self.__gstyle, gstyle)
        (_vstyle_n, _vstyle_w, _vstyle_c) = init_styles(self.__vstyle, vstyle)
        (_estyle_n, _estyle_w, _) = init_styles(self.__estyle, estyle)

        if vs is None:
            vs = set(self.vertices) # faster access

        vg = Digraph()

        # graph attributes
        sargs = { }
        for s in _gstyle_n:
           val = _gstyle_n[s](self) if callable(_gstyle_n[s]) else _gstyle_n[s]
           if val is not None:
               if s in _gstyle_w:
                   val = _gstyle_w[s](self, val)
               if val is not None:
                   sargs[s] = val
        vg.attr('graph', **sargs)

        ##### cluster support (graph)
        nclusters = _gstyle_c.get('nclusters', 0)
           ## set cluster attribute
        if nclusters > 0 and 'cluster' in _gstyle_c:
            cattr_fn = _gstyle_c['cluster']
            for c in range(0, nclusters):
                ret = cattr_fn(self, c)
                if type(ret) != dict:
                    raise RuntimeError("expecting cluster attributes inside a dictionary!")
                with vg.subgraph(name="cluster_%d" % c) as c:
                    c.attr(**ret)

        # vertex attributes
        for v in vs:
            sargs = { }
            for s in _vstyle_n:
                val = _vstyle_n[s](self, v) if callable(_vstyle_n[s]) else _vstyle_n[s]
                if val is not None:
                    if s in _vstyle_w:
                        val = _vstyle_w[s](self, v, val)
                    if val is not None:
                        sargs[s] = val
            vg.node(str(v), **sargs)

            # cluster support
            if nclusters > 0 and 'cluster' in _vstyle_c:
                c = _vstyle_c['cluster'](self, v)
                if c is not None:
                   with vg.subgraph(name='cluster_%d' % int(c)) as c:
                       c.node(str(v))

        # edge attributes
        if induced:
            for e in self.edges:
                if e[0] in vs and e[1] in vs:
                    sargs = { }
                    for s in _estyle_n:
                        val = _estyle_n[s](self, e) if callable(_estyle_n[s]) else _estyle_n[s]
                        if val is not None:
                            if s in _estyle_w:
                                val = _estyle_w[s](self, e, val)
                            if val is not None:
                                sargs[s] = val

                    vg.edge(str(e[0]), str(e[1]), **sargs)

        if pipe:
            return (vg.pipe(format=format, **kwargs))
        else:
            if 'ipykernel' in sys.modules:
                return vg
            else:
                return vg.render(filename=filename, cleanup=True, format=format, **kwargs)

    def view(self, host='0.0.0.0', port='8888', viewer=None, **kwargs):
        if viewer is None:
            from .webview import WebView
            _viewer = WebView()
        else:
            _viewer = viewer

        _viewer.add_graph(self, **kwargs)

        if viewer is None:
            _viewer.run(host=host, port=port)
