import fnmatch
from colorama import Fore, Style
import clang.cindex as ci
import sys
import inspect
import re

from .cnode import CNode
from .std_cnodes import *
from .spec_cnodes import *

# cursor_kind => entity
# order matters
mapping_rules = [
    (r'^(ForStmt|WhileStmt|DoStmt)$', CNodeLoop)

]

def __kobj_to_entity(kobj):


    rules = {ci.CursorKind.TRANSLATION_UNIT: 'Module',
             ci.CursorKind.StmtExpr: "StmtExpr" }

    entity = rules.get(kobj, None)

    if entity is not None:
        return entity

    # XPTO_ABC_DEF => XptoAbcDef
    kind = kobj.name
    entity = ''.join([n.capitalize() for n in kind.split('_')])

    return entity

# this method maps the class name with entity
def __normalize_class_name(cls_name):
    if cls_name.startswith("CNode"):
        name = cls_name[5:].lower()
        if name:
            return name
        else:
            return "node"
    else:
        return cls_name

# returns tuple (ancestry (list of entities), interfaces (list of classes)) from entity
def __compute_entity_metadata(kobj):

    # rules:
    #     1. if there is a explicit class (cls) with the same name as entity
    #     2. match <mapping_rules> dict
    #     3. match default cls


    # check if there is a class that matches the entity name.
    # if we have a match, we are done
    kind = kobj.name
    entity = __kobj_to_entity(kobj)

    match_cls = [ cls for (name, cls) in inspect.getmembers(sys.modules[__name__], inspect.isclass) if name == entity]
    if match_cls:
        # 1.
        cls = match_cls[0]
    else:
        # we need to look at the mapping rules
        cls = None
        # 2.
        for rule in mapping_rules:
            if re.search(rule[0], entity) is not None:
                cls = rule[1]
                break
        if not cls:
            # default
            # 3.

            if kobj.is_declaration():
                cls = CNodeDecl
            elif kobj.is_statement():
                cls = CNodeStmt
            elif kobj.is_expression():
                cls = CNodeExpr
            else:
                cls = CNode

    ancestry  = [ __normalize_class_name(c.__name__) for c in cls.__mro__[0:-1]]

    if ancestry[0] != entity:
        ancestry.insert(0, entity)

    result = (cls, entity, ancestry)

    return result

# __entities: entity => (kind, ancestry, interfaces)
# __kinds: kind => entity
def __entity_map():
    __entities = {}
    __kinds = {}

    for k in ci.CursorKind.get_all_kinds():
        kind = k.name
        (cls, entity, ancestry) = __compute_entity_metadata(k)
        __entities[entity] = (cls, kind, ancestry)
        __kinds[kind] = entity
    return (__entities, __kinds)

(__entities, __kinds) = __entity_map()

# metadata: entity => (cls, kind, ancestry)
def metadata(entity):
    return __entities[entity]

def kind_to_entity(kind):
    return __kinds[kind]

def entities():
    return list(__entities.keys())

def info_entity(entity, filter, v):

    (cls, _, ancestry) = metadata(entity)

    chain = f"{Fore.LIGHTBLACK_EX} ⭅ ".join([f"{Fore.LIGHTBLACK_EX}%s{Style.RESET_ALL}" % e for e in ancestry[1:]])
    header = f"\n{Fore.LIGHTBLUE_EX}%s{Fore.LIGHTBLACK_EX} ⭅ {Style.RESET_ALL}%s" % (entity, chain)

    info = f"{header}\n{Fore.LIGHTBLUE_EX}%s{Style.RESET_ALL}\n" % ("=" * 60)

    # class => attribute name => doc
    attr_docs = {}
    attributes = set()
    entity_description = ""
    interfaces = cls.__mro__[0:-1]
    for c in interfaces:
            entity = __normalize_class_name(c.__name__)
            if entity_description == "" and c.__doc__ is not None:
                entity_description = c.__doc__

            keys = sorted(c.__dict__.keys())
            for attr in keys:
                if attr[0] != '_' and attr not in attributes:
                    if entity not in attr_docs:
                        attr_docs[entity] = {}

                    def signature2str(name, fn):

                        params = []
                        if callable(fn):
                                name = fn.__name__
                                sig = inspect.signature(fn)
                                iterate_first = False
                                for p in sig.parameters:
                                    if not iterate_first: # ignore first parameter
                                        iterate_first = True
                                    else:
                                        # (attribute name, optional, is kw variable, annotation )
                                        params.append((p,
                                                    sig.parameters[p].default != inspect.Parameter.empty,
                                                    sig.parameters[p].kind == inspect.Parameter.VAR_KEYWORD,
                                                    "" if sig.parameters[p].annotation == inspect.Parameter.empty else ":%s" % sig.parameters[p].annotation.__name__))

                        if params:
                            sig = f"{name}(%s)" % ", ".join([ ("..." if p[2] else f"[{p[0]}{p[3]}]" if p[1] else f"{p[0]}{p[3]}") for p in params])

                        else:
                            sig = f"{name}"

                        return sig

                    attr_docs[entity][attr] = (signature2str(attr, c.__dict__[attr]), c.__dict__[attr].__doc__)
                    attributes.add(attr)

    info = info + f"{Fore.LIGHTGREEN_EX}{entity_description}{Style.RESET_ALL}\n"

    for c in ancestry:
        if c != 'node' or v > 1:
            if c in attr_docs:
                for attr in attr_docs[c]:
                    if fnmatch.fnmatch(attr, filter):
                        desc =  attr_docs[c][attr][1]
                        sig =attr_docs[c][attr][0]
                        if desc is not None:
                            if v == 0:
                                # first line
                                desc = desc.lstrip().split('\n', 1)[0]
                            else:
                                desc = desc.lstrip()
                        else:
                            desc = "?"
                        info = info + f"- {Fore.LIGHTWHITE_EX}{sig} {Fore.LIGHTBLACK_EX}({c}) {Style.RESET_ALL}\n"
                        info = info + f"  {Fore.LIGHTYELLOW_EX}{desc}{Style.RESET_ALL}\n"

    return info


def to_cnode(cursor, attrs):


    entity = kind_to_entity(cursor.kind.name)

    (cls, _, ancestry) = metadata(entity)

    # default attributes passed from source_manager, such as 'ast', 'id', etc.
    _attrs = attrs.copy()
    _attrs.update({'entity': entity, 'ancestry': ancestry, 'pragmas': None, 'attributes': None})

    cnode = cls()

    if _attrs['module'] is None:
        _attrs['module'] = cnode


    cnode._init(_attrs, cursor)

    return cnode





