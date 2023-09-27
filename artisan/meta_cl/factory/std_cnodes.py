from .cnode import CNode

__all__ = ["CNodeDecl", "CNodeStmt", "CNodeExpr", "CNodeLoop", "CNodeOp"]

class CNodeDecl(CNode):
    """Declaration CNode"""

    @property
    def name(self):
        return self.__cursor__.spelling

    @property
    def is_local(self):
        return self.ancestor_search(fn=lambda node:node.isentity("FunctionDecl")) is not None

    @property
    def is_global(self):
        return not self.is_local

class CNodeStmt(CNode):
    pass

class CNodeExpr(CNode):
    pass


class CNodeLoop(CNodeStmt):
    pass

class CNodeOp(CNodeExpr):
    @property
    def symbol(self):
        raise RuntimeError("Method 'symbol' not implemented!")

