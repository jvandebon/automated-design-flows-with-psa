#!/usr/bin/env artisan

from meta_cl import *
from design_flow_patterns import *
from normalisation import *

class DesignFlow:
    def __init__(self, name, cxx_spec=None):
        self.ast = None
        self.data = {}
        self.flow = []
        self.name = name
        self.src = None
        self.dest = None
        self.cxx_spec = cxx_spec

    def add_pattern(self, pattern, params={}):
        self.flow.append((pattern, params))

    def branchpoint(self, ast, data, branch_decision=None, flows=[]):
        path_idxs = branch_decision(ast,data)
        for idx in path_idxs:
            path = flows[idx]
            path.run(self.src, self.dest+f"-{path.name}", ast=self.ast, data=self.data)

    def add_branchpoint(self, branch_decision, flows):
        self.flow.append((self.branchpoint, {'branch_decision': branch_decision, 'flows': flows}))

    def build_typedef_map(self):
        structs = self.ast.query("td{TypedefDecl}=>st{StructDecl}")
        self.data['struct_map'] = {}
        for row in structs:
            name = row.td.type.spelling
            self.data['struct_map'][name] = [(mem.type.spelling, mem.name) for mem in row.st.children]

    def normalise(self):
        scopify(self.ast)
        normalise_pointer_dereferences(self.ast)
        separate_vardecl_and_init(self.ast)
        inline_fns_with_loops(self.ast)
        self.ast.sync(commit=True)

    def run(self, src, dest, args={}, ast=None, data=None):

        if data:
            self.data = data.copy()

        if not ast:
            self.ast = Ast(src, cxx_spec=self.cxx_spec).clone(dest)
            self.src = src
            self.dest = dest
            self.normalise()
            self.build_typedef_map()
        else:
            self.ast = ast.clone(dest, cxx_spec=self.cxx_spec)
            self.dest = dest

        for arg in args:
            self.data[arg] = args[arg]

        print(f"Running design flow {self.name} with dest {self.dest} and src {self.src}")
        for pattern,params in self.flow:
            pattern(self.ast,self.data,**params)

        return self.ast

