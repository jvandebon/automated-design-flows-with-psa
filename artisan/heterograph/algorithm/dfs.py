import graph_tool.search as gs
import copy
from inspect import signature

class StopSearch(gs.StopSearch):
   pass

def dfs_paths(g, vs=None, pre=None, post=None, data=None):

   def dfs_path(g, vx, pre, post, path, data):
      if pre is not None:
         pre(g, vx, path, data)

      upath = path.copy()
      upath.append(vx)
      for c_vx in g.out_vx(vx):
         dfs_path(g, c_vx, pre, post, upath, data)

      if post is not None:
         post(g, vx, path, data)

   path = []
   if vs is None:
      vs = g.source

   for vx in vs:
      dfs_path(g, vx, pre=pre, post=post, path=path, data=data)

def get_paths(g, vs=None):
   class GetPath:
      @staticmethod
      def pre(g, vx, path, data):
         data.last_pre_vx = vx

      @staticmethod
      def post(g, vx, path, data):
         # leaf!
         if data.last_pre_vx == vx:
               data.paths.append(path+[vx])

      def __init__(self, g, vs=None):
         self.paths=[]
         dfs_paths(g, vs, pre=GetPath.pre, post=GetPath.post, data=self)

   gp=GetPath(g, vs)
   return gp.paths


def dfs_visit(g, vx, pre=None, post=None, inh=None):

   # def tree(g, vx):
   #     def pre(g, vx, inh):
   #         print('pre=', vx, "inh=", inh)
   #         return inh+1
   #     def post(g, vx, synth):
   #         print(f"   post: {vx} => {synth}")
   #         if not synth:
   #             synth = []
   #         return synth + [vx]
   #     dfs_visit(g, vx, pre=pre, post=post, inh=0)

   class Visitor(gs.DFSVisitor):
      # properties: pre is always invoked before visiting children
      #             post is always invoked after visiting children
      def __init__(self, g, pre, post, root_inh):
            self.g = g
            self.pre = pre
            self.post = post

            self.synth = None
            if self.post:
               sig = signature(self.post)
               if len(sig.parameters) == 3:
                  # vx -> synthesised data
                  self.synth = { }

            self.inh = None
            if self.pre:
               sig = signature(self.pre)
               if len(sig.parameters) == 3:
                  # vx -> produced data to be inherited
                  self.inh = { }
                  self.root_inh = root_inh
                  self.src_vx = None

      def tree_edge(self, e):
         if self.inh is not None:
            # stores last source
            self.src_vx = self.g.to_vx[int(e.source())]

      def discover_vertex(self, u):
         if self.pre:
            if self.inh is not None:
               if self.src_vx is not None:
                  inh = self.inh.get(self.src_vx, None)
               else:
                  inh = self.root_inh
               vx = self.g.to_vx[int(u)]
               ret = self.pre(g=self.g, vx=vx, inh=inh)
               self.inh[vx] = ret
            else:
               self.pre(g=self.g, vx=self.g.to_vx[int(u)])

      # all out-edges have been visited
      def finish_vertex(self, u):
         if self.post:
            vx = self.g.to_vx[int(u)]
            if self.synth is not None:
               synth_data = []
               for v_child in self.g.out_vx(vx):
                  synth_data.extend(self.synth.get(v_child, []))
               ret = self.post(g=self.g, vx=vx, synth=synth_data)
               self.synth[vx] = ret
            else:
               self.post(g=self.g, vx=vx)

   visitor = Visitor(g=g, pre=pre, post=post, root_inh=inh)

   g.check_vx(vx, verify=True)

   ivx = g.to_ivx[vx]
   gs.dfs_search(g=g.igraph, source=ivx, visitor=visitor)

   if visitor.synth:
      return visitor.synth[vx]
