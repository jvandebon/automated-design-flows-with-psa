from typing import ChainMap
from .qgraph import QGraph
from ..algorithm.dfs import dfs_visit, get_paths

class QueryVisitor:

    @staticmethod
    def check_depth(root_depth, vx_depth, depth_constr):
        dist = vx_depth-root_depth

        check_min = depth_constr[0] is None or dist >= depth_constr[0]
        check_max = depth_constr[1] is None or dist <= depth_constr[1]

        return check_min and check_max


    def find_match(self, g, qgraph, chain, qchain, path_check, root_depth, fd, gd):
        def post(g, vx, synth):
            nonlocal qgraph, prefix_vx, prefix_qvx, qvx, rest_path

            vx_depth = g[vx].depth

            # fd
            if (fd is None) or (prefix_vx is not None) or QueryVisitor.check_depth(root_depth, vx_depth, fd):
                # gd
                if (gd is None) or  QueryVisitor.check_depth(root_depth, vx_depth, gd):
                    if path_check(g, qgraph, (prefix_vx, vx), (prefix_qvx, qvx)):
                        if len(rest_path) != 0:
                            child_vs = g.out_vx(vx)
                            for c_vx in child_vs:
                                partial_match = self.find_match(g=g, qgraph=qgraph, chain=(vx, c_vx), qchain=(qvx, rest_path), path_check=path_check, root_depth=root_depth, fd=fd, gd=gd)

                                for pm in partial_match:
                                    pmatch={**pm}
                                    pmatch[qgraph.pmap[qvx]['id']]=vx
                                    synth.append(pmatch)
                        else:
                            synth.append({qgraph.pmap[qvx]['id']:vx})
            return synth

        (prefix_qvx, path) = qchain

        if len(path) == 0:
            return []

        qvx,*rest_path = path
        (prefix_vx, root_vx) = chain

        return dfs_visit(g=g, vx=root_vx, post=post)

    def run(self, g, select, vs, vx_args, eg_args, path_check, match_filter, fd=None, gd=None):
        # g is target graph
        # select is AQL query
        # vs is the list of root vx
        # vx_args and eg_args translate vertex/edge arguments from AQL into annotations in qgraph
        # match_filter verifies if is match is to be included or discarded
        # fd: int or (int, int) specifying distance between root and node
        # gd: int or (int, int) specifying distance between any node

        if fd is not None:
            if type(fd) == int:
                fd=(fd, fd)
            elif type(fd) != tuple or len(fd) != 2:
                raise RuntimeError("invalid fd: expecting an int or an (int, int)!")

        if gd is not None:
            if type(gd) == int:
                gd=(gd, gd)
            elif type(gd) != tuple or len(gd) != 2:
                raise RuntimeError("invalid gd: expecting an int or an (int, int)!")                

        self.g = g
        qgraph = QGraph(select=select, vx_args=vx_args, eg_args=eg_args)

         # create paths from qgraph
        paths = get_paths(g=qgraph)

        if vs is None:
            vs = g.source

        matches=[]

        for vx in vs:
            for path in paths:
                match=[ m for m in self.find_match(g=g, qgraph=qgraph, chain=(None, vx), qchain=(None, path), path_check=path_check, root_depth=g[vx].depth, fd=fd, gd=gd) if match_filter(g, qgraph, m) ]
                matches.extend(match)
        return (qgraph, matches)