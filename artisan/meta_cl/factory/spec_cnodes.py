from .std_cnodes import *

from ..utils import token_id

import itertools

__all__ = ["FunctionDecl",
           "CallExpr", "ForStmt",
           "WhileStmt", "DoStmt", "IfStmt",
           "DeclRefExpr",
           "UnaryOperator", "BinaryOperator", "CompoundAssignmentOperator"]

class FunctionDecl(CNodeDecl):
    @property
    def signature(self):

        class FnSignature:
            def __init__(self, type, params):
                self.rettype = type
                self.params = params

            def __repr__(self):
                params_str = ", ".join([ f"{p[0].spelling} {p[1]}" for p in self.params])
                return f"{self.rettype.spelling} ({params_str})"

        cursor = self.__cursor__
        table = self.query("fd{FunctionDecl}={1}>param{ParmDecl}", where=lambda fd: fd.id == self.id )

        params = []
        for r in table:
            param = r.param
            params.append((param.__cursor__.type, param.spelling))

        return FnSignature(self.__cursor__.result_type, params)

    @property
    def body(self):
        table = self.query("fn{FunctionDecl}={1}>body{CompoundStmt}", where=lambda fn: fn.id == self.id)
        if len(table) == 0:
            return None
        else:
            return table[0].body

class CallExpr(CNodeExpr):
    @property
    def name(self):
        return self.spelling

    @property
    def is_operator(self):
        return self.name.startswith("operator")

    @property
    def args(self):
        # TODO: support operator calls!
        if self.is_operator:
            raise RuntimeError("operator calls not supported...")

        args = self.children[1:]

        return args

class ForStmt(CNodeLoop):
    def __verify_structure(self):
        if self.num_children != 4:
            raise RuntimeError(f"invalid for-loop structure at {self.location}!")

    @property
    def init(self):
        self.__verify_structure()
        return self.child(0)

    @property
    def condition(self):
        self.__verify_structure()
        return self.child(1)


    @property
    def increment(self):
        self.__verify_structure()
        return self.child(2)

    @property
    def body(self):
        self.__verify_structure()
        return self.child(3)


class WhileStmt(CNodeLoop):
    def __verify_structure(self):
        if self.num_children != 2:
            raise RuntimeError(f"invalid while loop structure at {self.location}!")

    @property
    def condition(self):
        self.__verify_structure()
        return self.child(0)

    @property
    def body(self):
        self.__verify_structure()
        return self.child(self.num_children-1)





class DoStmt(CNodeLoop):
    def __verify_structure(self):
        if self.num_children != 2:
            raise RuntimeError(f"invalid do-while loop structure at {self.location}!")

    @property
    def condition(self):
        self.__verify_structure()
        return self.child(1)

    @property
    def body(self):
        self.__verify_structure()
        return self.child(0)


class IfStmt(CNodeStmt):

    @property
    def condition(self):
        return self.child(0)

    @property
    def body(self):
        return self.child(1)

    @property
    def elsebody(self):
        if self.num_children == 3:
            return self.child(2)
        else:
            return None

class DeclRefExpr(CNodeExpr):
    @property
    def name(self):
        return self.spelling

    @property
    def decl(self):
        type = self.type
        if type is None:
            return None

        decl_cursor = self.__cursor__.referenced

        if decl_cursor is None:
            return None

        decl_cnode = self.ast.source_manager.get_cnode(decl_cursor)

        return decl_cnode


    @property
    def access(self):
        class AccessType:
            def __init__(self, ref, is_used, is_def):
                self.is_used = is_used
                self.is_def = is_def
                self.ref = ref
            def __repr__(self):
                return f"({self.ref.name}:{self.ref.location} - READ:{self.is_used}, WRITE:{self.is_def})"       

        ret = self.ancestor_search(fn=lambda node: (node.isentity("op") and node.symbol not in ['&', '*']) or node.isentity("DeclStmt"))
        if ret is None:
            return None # somehow we missed this case
         
        anchor = ret[-1]

        # check case ++ or -- 
        if anchor.isentity("UnaryOperator") and anchor.symbol in ["++", "--"]:
            return AccessType(self, True, True)
        
        # check if it is inside an array subscript
        ret2 = self.ancestor_search(fn=lambda node: node.isentity("ArraySubscriptExpr"), until_first_find=False)
        if ret2:
            _array = ret2[-1]
            if _array.num_children != 2:
                raise RuntimeError(f"internal error: assuming ArraySubscriptExpr has 2 children, but ended up having {_array.num_children}!")

            # index 1 is the subscript expression            
            res = _array.child(1).query("ref{DeclRefExpr}", where=lambda ref: ref.id == self.id)

            if res: # inside subscript
                return AccessType(self, True, False)


        if anchor.isentity("DeclStmt"):
            return AccessType(self, True, False)
        else:
            op_children = anchor.children

            if len(op_children) == 2:
                if anchor.symbol == '=' or anchor.isentity('CompoundAssignmentOperator'):
                    # op_children[0] in ret, checks if reference is somewhere in LHS
                    if self == op_children[0] or op_children[0] in ret: # LHS
                        if anchor.symbol == '=':
                            check=anchor.ancestor_search(fn=lambda n: n.isentity('CompoundAssignmentOperator') or (n.isentity('BinaryOperator') and n.symbol == '='))
                            if check is None:
                                return AccessType(self, False, True)
                            else:
                                return AccessType(self, True, True)
                        else:
                            # compound
                            return AccessType(self, True, True)
                    elif self == op_children[1] or op_children[1] in ret: # RHS
                        return AccessType(self, True, False)
                    else:
                        raise RuntimeError("internal error: we should not arrive at this point.")
                else:
                    # no assignment op
                    return AccessType(self, True, False)
            else:
                return None

class UnaryOperator(CNodeOp):
    @property
    def symbol(self):
        n=len(self.children)
        if n != 1:
            raise RuntimeError(f"invalid binary operator, expecting 1 child, found {n}!")

        tokens = [t for t in self.tokens]
        child_token0 = next(self.children[0].tokens)

        if token_id(tokens[0]) == token_id(child_token0):
            # operator is postfix
            return tokens[-1].spelling
        else:
            # operator is prefix
            return tokens[0].spelling

class BinaryOperator(CNodeOp):
    @property
    def symbol(self):
        n=len(self.children)
        if n != 2:
            raise RuntimeError(f"invalid binary operator, expecting 2 children, found {n}!")
        ltokens = len([t for t in self.children[0].tokens])
        t = next(itertools.islice(self.tokens, ltokens, None))
        return t.spelling

class CompoundAssignmentOperator(CNodeOp):
    @property
    def symbol(self):
        n=len(self.children)
        if n != 2:
            raise RuntimeError(f"invalid binary operator, expecting 2 children, found {n}!")
        ltokens = len([t for t in self.children[0].tokens])
        t = next(itertools.islice(self.tokens, ltokens, None))
        return t.spelling








