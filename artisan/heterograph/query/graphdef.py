class QueryGraphDef:
    def __init__(self, src=None, snk=None, steps=None):
        if src is None:
            src = set()
        if snk is None:
            snk = set()
        if steps is None:
            steps = []
        self.src = src
        self.snk = snk
        self.steps = steps

    def __repr__(self):
        return "%s (src: %s, snk: %s, steps: %s)" % (self.__class__.__name__, self.src, self.snk, self.steps)

    @staticmethod
    def args_label(args):
        sep = ", " if len(args[0]) != 0 and len(args[1]) != 0 else ""
        return "(%s%s%s)" % (str(args[0])[1:-1], sep, str(args[1])[1:-1])

    def build(self, qgraph, vx_args, eg_args):
        ids = qgraph.pmap['ids']
        for s in self.steps:
            # 2-tuple: vertex (<vertex-id>, v-args)
            # 3-tuple: edge (e-args, <vertex-id-src>, <vertex-id-target>)
            if type(s) == tuple:
                len_s = len(s)
                if len_s == 2:
                    # vertex

                    # only add if vertex does not exist
                    vertex_id = s[0]
                    if vertex_id not in ids:
                       v = qgraph.add_vx(1)
                       ids[vertex_id] = v
                       qgraph.pmap[v]['id'] = vertex_id
                       qgraph.pmap[v]['args'] = ""
                    else:
                        v = ids[vertex_id]

                    vargs = s[1]
                    if vargs is not None:
                        if qgraph.pmap[v]['args'] == "":
                            vx_args(qgraph, v, *vargs[0], **vargs[1])
                            qgraph.pmap[v]['args'] = QueryGraphDef.args_label(vargs)
                        else:
                            raise RuntimeError("arguments for vertex [%s] have already been supplied!" % vertex_id)
                    else:
                        vx_args(qgraph, v) # default
                    continue
                elif len_s == 3:
                    eargs = s[0]
                    vertex_id_s = s[1]
                    vertex_id_t = s[2]
                    edge = (ids[vertex_id_s], ids[vertex_id_t])

                    # only add if edge does not exist
                    if not qgraph.check_edge(edge):
                        qgraph.add_edge(ids[vertex_id_s], ids[vertex_id_t])[0]
                        qgraph.pmap[edge]['args'] = ""

                    if eargs is not None:
                        if qgraph.pmap[edge]['args'] == "":
                            eg_args(qgraph, edge, *eargs[0], **eargs[1])
                            qgraph.pmap[edge]['args'] = QueryGraphDef.args_label(eargs)
                        else:
                            raise RuntimeError("arguments for edge (%s, %s) have already been supplied!" % (vertex_id_s, vertex_id_t))
                    else:
                        eg_args(qgraph, edge) # default


                    continue
            raise RuntimeError("invalid step: %s" % s)

        return qgraph
