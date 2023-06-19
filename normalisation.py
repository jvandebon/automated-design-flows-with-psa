#!/usr/bin/env artisan

from meta_cl import *
from util import *

## NORMALISATION
def add_scope(node):
    node.instrument(Action.before, code="{")
    node.instrument(Action.after, code=";\n}")
    node.instrument(Action.remove_semicolon, verify=False)

def scopify(ast):
    # scopify loops
    wave = 1
    while True:
        table = ast.query("forloop{ForStmt}", 
                    where=lambda forloop: not forloop.body.isentity("CompoundStmt"))
        if not table:
            break
        disjoint_table = table.apply(RSet.disjoint, target='forloop')
        for row in disjoint_table:
            add_scope(row.forloop.body)
        ast.commit()
        wave += 1
    # scopify conditionals
    wave = 1
    while True:
        table = ast.query("ifstmt{IfStmt}", 
                    where=lambda ifstmt: not ifstmt.body.isentity("CompoundStmt") or (ifstmt.elsebody and not ifstmt.elsebody.isentity("CompoundStmt")))
        if not table:
            break
        disjoint_table = table.apply(RSet.disjoint, target='ifstmt')
        for row in disjoint_table:
            if not row.ifstmt.body.isentity("CompoundStmt"):
                add_scope(row.ifstmt.body)
            elif not row.ifstmt.elsebody.isentity("CompoundStmt"):
                add_scope(row.ifstmt.elsebody)
        ast.commit()
        wave += 1

def separate_vardecl_and_init(ast):
    result = ast.query('ds{DeclStmt} ={1}> vd{VarDecl}', 
                        where=lambda ds, vd: (len(vd.children) > 0 
                                and not (len(vd.children) == 1 
                                and vd.children[0].isentity('TypeRef')) 
                                and not ds.parent.isentity('ForStmt') 
                                and not 'const' in vd.type.spelling
                                and not 'auto' in vd.unparse()[0:5]
                                and '=' in vd.unparse()))
    for row in result:
        if row.vd.name == row.vd.children[-1].unparse() or '{0}' in row.vd.children[-1].unparse():
            # HACK: catches a case std::ofstream ofile; -- tries to set ofile = ofile;
            # TODO: debug issue with = {0}
            continue
        new_decl = "%s;\n" % row.vd.unparse()[:row.vd.unparse().index('=')]
        new_init = "%s = %s;" % (row.vd.name, row.vd.children[-1].unparse())
        row.ds.instrument(Action.replace, code="%s%s\n" % (new_decl, new_init))
    ast.commit()

def find_scope(node):
    while node and not node.isentity('CompoundStmt'):
        node = node.parent
    return node

def normalise_pointer_dereferences(ast):
    results = ast.query('uop{UnaryOperator} ={1}> p{ParenExpr}', where=lambda uop: uop.symbol == '*')
    for row in results:
        ## ASSUMES FORMAT: *(POINTER + exp...)
        refs = row.p.query('ref{DeclRefExpr}', where=lambda ref: pointer_type(ref.type))
        if len(refs) != 1:
            print("Cannot normalise pointer dereference at %s, exiting." % row.uop.location)
        pointer = refs[0].ref
        row.uop.instrument(Action.replace, code="%s[%s]"%(pointer.name, row.p.unparse().replace(pointer.name, '0')))
    ast.commit()

def inline_fns_with_loops(ast):
    # find function calls inside loops
    calls = set([row.c for row in ast.query('l{ForStmt} => c{CallExpr}')])
    for call in calls:
        # check if functions are defined locally, and if they have inner loops
        fn = ast.query("fn{FunctionDecl}=>l{ForStmt}", where=lambda fn: fn.name == call.name and fn.body)
        if not fn:
            continue
        inline_fn(ast,fn[0].fn, call)
    ast.sync(commit=True)

