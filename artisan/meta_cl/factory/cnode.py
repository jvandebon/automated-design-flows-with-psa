from colorama import Fore, Style
import inspect
import copy
import clang.cindex as ci

from heterograph.algorithm.dfs import dfs_visit, StopSearch
from heterograph.utils.display import terminal_cpp
from heterograph.utils.notebook import is_notebook

from ..unparser import pretty_print, unparse
from ..query import query_visitor
from ..utils import token_id, shape_type
from ..instrumentation import CodeInstrument
from ..action import Action

class CNode:
    class Location:
        def __init__(self, srcname, line, column):
            self.srcname = srcname
            self.line = line
            self.column = column

        def __repr__(self):
            if self.line == 0 and self.column == 0:
                return f"{self.srcname}"
            else:
                return f"{self.srcname}:{self.line}:{self.column}"


    class Range:
        def __init__(self, srcname, line_start, column_start, line_end, column_end):
            self.srcname = srcname
            self.line_start = line_start
            self.column_start = column_start
            self.line_end = line_end
            self.column_end = column_end

        def __repr__(self):
            if self.line_start == self.line_end:
                return f"{self.srcname}:{self.line_start}:[{self.column_start}➼{self.column_end}]"
            else:
                return f"{self.srcname}:{self.line_start}:{self.column_start} ⟼  {self.line_end}:{self.column_end}"

    def _init(self, attrs, cursor):
        class Attrib: pass
        self.__attrs__ = Attrib()
        for attr in attrs:
            setattr(self.__attrs__, attr,  attrs[attr])
        setattr(self, '__cursor__', cursor)

    @property
    def id(self):
        return self.__attrs__.id

    @property
    def entity(self):
        return self.__attrs__.entity

    @property
    def ancestry(self):
        return self.__attrs__.ancestry

    @property
    def depth(self):
        return self.__attrs__.depth

    @property
    def ast(self):
        return self.__attrs__.ast

    @property
    def module(self):
        return self.__attrs__.module


    @property
    def srcname(self):
        return self.__attrs__.srcname

    @property
    def pragmas(self):
        return self.__attrs__.pragmas

    @property
    def attributes(self):
        return self.__attrs__.attributes

    @property
    def location(self):
        clang_location = self.__cursor__.location
        return CNode.Location(self.srcname, clang_location.line, clang_location.column)

    @property
    def range(self):
        clang_extent = self.__cursor__.extent
        range = CNode.Range(self.srcname, clang_extent.start.line, clang_extent.start.column,
                                          clang_extent.end.line, clang_extent.end.column)
        return range

    @property
    def tag(self):
        loc = self.location
        return "%s_%s_%d_%d" % (self.srcname.replace(".", "_").replace("/", "_"), self.entity.lower(), loc.line, loc.column)

    @property
    def parent(self):
        g = self.ast
        _in = g.in_vx(self.id)
        if len(_in) == 0:
            return None
        else:
            return g[_in[0]]

    @property
    def type(self):
        if self.__cursor__.type.kind == ci.TypeKind.INVALID:
            return None
        else:
            return self.__cursor__.type

    @property
    def shape(self):
        return shape_type(self.type)            

    @property
    def num_children(self):
        g = self.ast
        return g.num_out_vx(self.id)

    def child(self, idx):

        if idx >= self.num_children:
            raise IndexError(f"child index '{idx}' exceeds number of children ({self.num_children})!")
        g = self.ast
        _out = g.out_vx(self.id)

        return g[_out[idx]]

    @property
    def children(self):
        g = self.ast
        _out = g.out_vx(self.id)
        if len(_out) == 0:
            return []
        else:
            return [ g[vx] for vx in _out ]

    @property
    def ancestors(self):
        ret = []
        parent = self.parent
        while parent:
            ret.append(parent)
            parent = parent.parent
        return ret

    # until_first_find == True: returns list of ancestors until finding element
    #                     else: return list of ancestors until last element

    def ancestor_search(self, fn, until_first_find=True):
        parent = self.parent
        n = 0
        ret = []
        pos = 0
        last_find_pos = None
        while parent:            
            ret.append(parent)
            if fn(parent):
                last_find_pos = pos
                if until_first_find:
                    break
                    
            parent = parent.parent
            pos = pos + 1

        return None if last_find_pos is None else ret[0:last_find_pos+1]



    @property
    def descendants(self):
        root = self.id
        def collect_child(g, vx, synth):
            nonlocal root
            if root != vx:
                synth.append(g[vx])
            return synth
        children = dfs_visit(g=self.ast, vx=self.id, post=collect_child)
        return children

    def is_outermost(self, context=None):
        if context is None:
            context = lambda node: node.entity == self.entity
        parent = self.parent
        while parent:
            if context(parent):
                return False
            parent = parent.parent
        return True

    def is_innermost(self, context=None):
        if context is None:
            context = lambda node: node.entity == self.entity

        for vx in self.descendants:
            if context(vx):
                return False
        return True

    def rank(self, nodes=None, context=None):
        if context is None:
            # rank all children
            context = lambda node: True

        # level => counter
        counters = { }
        ranks = {  }

        def pre(g, vx, inh):
            nonlocal context, ranks

            if context(g[vx]):
                level = len(inh) + 1
                if level not in counters:
                    counters[level] = 0
                else:
                    counters[level] = counters[level] + 1
                inh_ret = inh + [counters[level]]
                if (nodes is None) or (vx in nodes):
                   ranks[vx] = copy.copy(inh_ret)
            else:
                inh_ret = inh

            return inh_ret

        dfs_visit(g=self.ast, vx=self.id, pre=pre, inh=[])

        return ranks


    def isentity(self, name):
        return name in self.ancestry


    def view(self, host='0.0.0.0', port='8888', **kwargs):
        g = self.ast

        vx = self.id

        def collect_child(g, vx, synth):
            synth.append(vx)
            return synth

        children = dfs_visit(g=g, vx=vx, post=collect_child)

        g.view(host=host, port=port, vs=children, **kwargs)

    @property
    def spelling(self):
        return self.__cursor__.spelling

    # replace is a dict {node: 'code-replacement'}
    def unparse(self, changes=False, align=False, replace=None):
        ast = self.ast
        if replace:
            if changes:
                instr = ast.instr.clone()
            else:
                instr = CodeInstrument()            
            for r in replace:    
                instr.apply(cnode=r, action=Action.replace, code=replace[r])
            code = unparse(self, instrument=instr)
        elif changes:
           code = unparse(self, instrument=ast.instr)
        else:
           code = unparse(self)

        if align:
            code = pretty_print(code)

        # BUG: is_notebook() returns True when importing hls4ml
        # if is_notebook():
        #     code = terminal_cpp(code)
        
        return code

    def instrument(self, action, **kwargs):
        ast = self.ast
        ast.instr.apply(self, action=action, **kwargs)

    def query(self, select, **kwargs):
        ast = self.ast
        return query_visitor(ast, vs=[self.id], select=select, **kwargs)

    def dfs_visit(self, pre=None, post=None, inh=None):
        ast = self.ast
        dfs_visit(ast, self.id, pre=pre, post=post, inh=inh)


    def tree(self, info=None, depth=-1):
        def pre(g, vx, inh):

            if depth != -1 and inh > depth:
                return inh

            self = g[vx]
            if inh > 0:
                indent = " " * (3*inh) + "├─"
            else:
                indent = ""

            if info:
                info_str = info(self)
            else:
                info_str = str(self.range)

            print(f"{Fore.LIGHTRED_EX}{indent} {self.id}:{Fore.LIGHTBLUE_EX }{self.entity}{Fore.LIGHTBLACK_EX} {info_str}{Style.RESET_ALL}")

            return inh+1


        self.dfs_visit(pre=pre, inh=0)

    def overlaps(self, node)->bool:
        self_tokens = set([token_id(t) for t in self.tokens ])
        node_tokens = set([token_id(t) for t in node.tokens ])

        intersect = self_tokens.intersection(node_tokens)

        return len(intersect) != 0

    def encloses(self, node)->bool:
        self_tokens = set([token_id(t) for t in self.tokens ])
        node_tokens = set([token_id(t) for t in node.tokens ])

        return self_tokens.issuperset(node_tokens)

    @property
    def tokens(self):
        return self.__cursor__.get_tokens()

    def help(self, filter:str="*", v:int=0):
        from .engine import info_entity
        info = info_entity(self.entity, filter, v)
        print(info)

















