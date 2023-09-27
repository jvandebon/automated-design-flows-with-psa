import os
import os.path as osp
import sys

import clang.cindex as ci
from colorama import Fore, Style

from . import utils
from . import cmdline

from .preprocessor import preprocess
from .unparser import pretty_print
from .factory.engine import to_cnode
from collections import OrderedDict
import shlex

class CLSource:
    def __init__(self, name):
        self.name = name
        self.cnodes = None
        self.code = None

        # clang translation unit, used for
        # indexing cnodes by Clang cursors
        self.translation_unit = None

    @property
    def module(self):
        if self.cnodes is None:
            return None
        else:
            # first element of cnodes is always
            # a module (translation unit cursor)
            return self.cnodes[0]

    def __repr__(self):
        return "(%s, %d cursor%s)" % (self.name,
                                      len(self.cnodes) if self.cnodes is not None else 0,
                                      "" if (self.cnodes is not None and len(self.cnodes) == 1) else "s")

class CLSources:
    @staticmethod
    def prepare(srcnames: list, workdir):
        pathnames = [ osp.abspath(s) for s in srcnames ]

        # remove dupes
        pathnames = list(OrderedDict.fromkeys(pathnames))
        for p in pathnames:
            if not osp.exists(p):
                raise FileNotFoundError("cannot find source file %s!" % p)

        wd0 = osp.commonpath([osp.dirname(p) for p in pathnames])

        if workdir is not None:
            wduser = osp.abspath(workdir)
            if not osp.exists(wduser):
                raise RuntimeError(f"workdir cannot be found: {workdir}!")
            if osp.commonpath([wduser]) != osp.commonpath([wduser, wd0]):
                raise RuntimeError(f"specified workdir '{workdir}' does not contain sources in '{wd0}'!")
            _workdir = wduser

        else:
            _workdir = wd0

        if _workdir == "/":
            raise RuntimeError("invalid project dir: root ('/')!")

        srcs = [ CLSource(osp.relpath(p, _workdir)) for p in pathnames]

        return (_workdir, srcs)

    def __init__(self, srcnames:list, workdir):

        if len(srcnames) == 0:
            raise RuntimeError("no source-file names specified!")

        #db = [ (srcname, <CLSource>), ... ]
        (self.workdir, self.db) = CLSources.prepare(srcnames, workdir)

        # index_src: srcname => n
        n = 0
        self.idx_src = { }
        for s in self.db:
            self.idx_src[s.name] = n
            n = n + 1

        # index: vx => cnode
        # this is used in ast[id] to get the cnode
        self.idx_vx = { }

        # index: tuc => cursor_hash => cnode
        # this is used to convert a cursor to a cnode
        self.idx_cursor = { }

        # counter for iterator (self.__next__)
        self.__i = 0

    def set_module(self, srcname, tu, cnodes, code):

        db_elem = self[srcname]
        if db_elem.cnodes is not None:
            raise RuntimeError(f"must clear module: {srcname}!")
        db_elem.cnodes = cnodes
        db_elem.code = code
        db_elem.translation_unit = tu

        # update indices
        index = { }
        for cnode in cnodes:
            self.idx_vx[cnode.id] = cnode
            cursor_hash = cnode.__cursor__.hash
            index[cursor_hash] = cnode

        #if len(index) != len(cnodes):
        #    raise RuntimeError(f"internal error: invalid cnode indexing: indexed {len(index)} nodes, expected {len(cnodes)}!")

        self.idx_cursor[tu] = index


    def clear_module(self, srcname):
        db_elem = self[srcname]

        for cnode in db_elem.cnodes:
            del self.idx_vx[cnode.id]

        del self.idx_cursor[db_elem.translation_unit]

        db_elem.cnodes = None
        db_elem.code = None
        db_elem.translation_unit = None

    def __getitem__(self, item):
        if type(item) == int:
            return self.db[item]
        if type(item) == str:
            db_n = self.idx_src.get(item, None)
            if db_n is None:
                raise RuntimeError("invalid source file: %s" % item)
            return self.db[db_n]
        if type(item) == ci.Cursor:
            source = item.source
            return self[source]
        raise RuntimeError("invalid source: %s!" % str(item))

    def __len__(self):
        return len(self.db)

    def __iter__(self):
        return self

    def __next__(self):
        if self.__i < len(self):
            ret = self.db[self.__i]
            self.__i += 1
            return ret
        else:
            self.__i = 0
            raise StopIteration


class CLSourceManager:
    DIAGNOSTIC_SEVERITY = ['Ignored', 'Note', 'Warning', 'Error', 'Fatal']

    def __init__(self, ast, cmd, *, workdir, cxx_spec):
        self.ast = ast

        (srcnames, args, defs) = cmdline.parse(cmd)

        self.args = args
        self.defs = defs
        self.parse_options = ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        self.index = ci.Index.create(excludeDecls=True)
        self.cxxflags_spec = CLSourceManager.__set_cxxflags_spec(cxx_spec)
        self.sources = CLSources(srcnames, workdir)

        # fallback: empty string means associate to TranslationUnit
        self.__pragma_rules = [ lambda pragma: "" ]

    def __set_cxxflags_spec(spec):
        if spec is None:
            return []

        spec_var = 'META_CL_CXXFLAGS_'+spec.upper()
        spec_fe_var = 'META_CL_CXXFLAGS_FE_'+spec.upper()

        ret = []

        for var in [spec_var, spec_fe_var]:
            flags = os.getenv(var)
            if flags is not None:
                ret.extend(shlex.split(flags))

        if ret == []:
            raise RuntimeError(f"CXX flags spec '{spec}' not found! Expecting: env variable '{spec_var}' or '{spec_fe_var}!")

        return ret

    @property
    def workdir(self):
        return self.sources.workdir

    def get_cnode(self, cursor):
        tu = cursor.translation_unit
        if tu not in self.sources.idx_cursor:
            raise RuntimeError(f"invalid cursor - translation unit not registered in this AST!")

        index = self.sources.idx_cursor[tu]

        hash = cursor.hash

        if hash not in index:
            raise  RuntimeError(f"cannot translate cursor into a cnode!")

        cnode = index[hash]

        return cnode

    def build_module(self, *, srcname, preprocessor_control=None):

        source = self.sources[srcname]

        if source.module is None: # first time load
            # 1. parsing code phase
            srcname_f = osp.join(self.workdir, source.name)
            print(f"    +--- {Fore.YELLOW}ast module{Fore.LIGHTBLACK_EX} ◀◀◀ {Fore.LIGHTCYAN_EX}{source.name}{Fore.LIGHTBLACK_EX} ({srcname_f}){Style.RESET_ALL}")

            try:
                with open(srcname_f, 'r') as f:
                    code = f.read()

            except FileNotFoundError as e:
                print(f"   => cannot find source '{srcname_f}' in '{self.workdir} ")
                sys.exit(-1)

            # we are getting the source for the first time,
            # we must preprocess it first
            code=preprocess(code=code, defs=self.defs, args=self.args, control=preprocessor_control)
        else:
            nchanges = self.ast.instr.num_module_changes(source.translation_unit)
            print(f"    +--- {Fore.YELLOW}ast module {Fore.LIGHTCYAN_EX}({source.name}){Fore.LIGHTBLACK_EX} ◀◀◀ {Fore.LIGHTBLACK_EX}{nchanges} change(s){Style.RESET_ALL}")
            tuc=source.module
            code = tuc.unparse(changes=True, align=True)
            code=preprocess(code=code, defs=self.defs, args=self.args, control=preprocessor_control)

            self.remove_module(source.name)


        _cwd = os.getcwd()
        os.chdir(self.workdir)

        tu = ci.TranslationUnit.from_source(filename=srcname,
                                            unsaved_files=[(srcname, code)],
                                            options=self.parse_options,
                                            args=self.args + self.defs + self.cxxflags_spec,
                                            index=self.index)
        os.chdir(_cwd)

        # 2. checking for errors
        try:
            if tu.diagnostics:
                has_errors = False
                for diag in tu.diagnostics:
                    if diag.severity >= 3:
                        print (f"   => {Fore.BLUE}{CLSourceManager.DIAGNOSTIC_SEVERITY[diag.severity]}: {diag.category_name}{Fore.RED}"
                               f" [{utils.clang_location_str(diag.location, self.workdir)}]: {Fore.YELLOW}{diag.spelling}{Style.RESET_ALL}")
                    if (not has_errors) and diag.severity >= 3:
                        has_errors = True
                if has_errors:
                    raise ci.TranslationUnitLoadError("compilation error")
        except ci.TranslationUnitLoadError as e:
            print("[x] exiting with errors!", file=sys.stderr)
            sys.exit(-1)

        # 3. cnodes build
        self.__create_module_cnodes(srcname, tu, code)

    def __create_module_cnodes(self, srcname, tu, code):
        # this method visits clang AST (only nodes that have location in the source)

        # parent is a cursor
        def visit(g, cursor, parent_cursor=None, parent_vx=None, depth=0):

            if cursor.kind == ci.CursorKind.TRANSLATION_UNIT or ci.conf.lib.clang_Location_isFromMainFile(cursor.location):
                # disabled C++ templates - there is an issue of replication when instantiating declaration
                # check oneapi/kmeans.cpp
                # if cursor.kind == ci.CursorKind.CLASS_TEMPLATE:
                #     raise RuntimeError("C++ templates are not yet suppported.")

                vx = None
                if cursor.kind != ci.CursorKind.UNEXPOSED_EXPR: # we ignore nodes that are unexposed
                    vx = ast.add_vx()

                    # additional attributes
                    tuc = None if parent_vx is None else cnodes[0]
                    attrs={'ast': ast, 'id': vx, 'depth': depth, 'module': tuc, 'srcname': srcname}

                    cnode = to_cnode(cursor=cursor, attrs=attrs)

                    cnodes.append(cnode)

                    if parent_vx is not None:
                        ast.add_edge(parent_vx, vx)

                    _parent_cursor = cursor
                    _parent_vx = vx
                    _depth = depth + 1


                else:
                    _parent_cursor = parent_cursor
                    _parent_vx = parent_vx
                    _depth = depth


                # traverse children of this node
                for child_cursor in cursor.get_children():
                    visit(ast, child_cursor, _parent_cursor, _parent_vx, _depth)


        ast = self.ast

        cnodes = []
        ast.read_only = False
        visit(self, tu.cursor)
        ast.read_only = True

        self.sources.set_module(srcname, tu, cnodes, code)


    def remove_module(self, srcname):
        ast = self.ast

        source = ast.sources[srcname]

        module = source.module
        tu = module.__cursor__.translation_unit

        # remove vertices from graph
        ast.read_only = False
        ast.remove_subgraph(module.id)
        ast.read_only = True

        # removes all cnodes from sources
        self.sources.clear_module(srcname)

        # deallocate clang tu
        del tu

    def parse_pragmas(self, rules):

        # an example of a pragma rule
        # def artisan_pragmas(pragma):
        #     if pragma[0] == "artisan" and pragma[1] == "top":
        #         return "FunctionDecl"
        #     return None
        # ast.parse_pragmas([artisan_pragmas])

        for source in self.sources:
            # step 1: extract pragmas
            pragmas = []

            if source.module is None:
                continue

            tokens = source.module.__cursor__.get_tokens()
            prev_line_num = -1
            prev_token_is_directive = False
            prev_token_is_pragma = False
            is_part_of_pragma = False
            prev_token = None
            pragma = None
            token_ini = None
            token_end = None
            for t in tokens:
                line_num = t.location.line
                is_new_line = line_num != prev_line_num
                is_directive = is_new_line and t.kind == ci.TokenKind.PUNCTUATION and t.spelling == "#"
                is_pragma = prev_token_is_directive and t.kind == ci.TokenKind.IDENTIFIER and t.spelling == "pragma"
                if is_pragma:
                    token_ini = prev_token
                is_part_of_pragma = prev_token_is_pragma or (is_part_of_pragma and prev_line_num == line_num)

                if is_part_of_pragma:
                    if pragma is None:
                        pragma = []
                    pragma.append(t.spelling)
                    token_end = t
                else:
                    if pragma is not None:
                        pragmas.append((pragma, token_ini, token_end))
                        token_ini = None
                        token_end = None
                        pragma = None

                #print(is_part_of_pragma, t.kind, t.spelling, t.location.line, t.location.column)
                prev_line_num = line_num
                prev_token_is_directive = is_directive
                prev_token_is_pragma = is_pragma
                prev_token = t

            # step 2: iterate pragmas, associate to cnode according to rules
            _rules = []
            _rules.extend(self.__pragma_rules) # default rule list: TBD
            if rules:
                _rules.extend(rules)

            for pragma in pragmas:
                cnode = None
                for rule in reversed(_rules): # priority: last element first
                    entity = rule(pragma[0])
                    if entity is not None:
                        if type(entity) == bool:
                            if entity:
                                raise RuntimeError("Invalid return: 'True'!")
                            else:
                                break
                        elif entity == "": # Module
                            cnode = source.module
                        else:
                            pragma_line_num = pragma[1].location.line
                            # not efficient, should use an index to speed it up
                            for cn in source.cnodes:
                                # ignore preprocessing
                                if cn.__cursor__.kind.value not in [ci.CursorKind.PREPROCESSING_DIRECTIVE.value,
                                                                    ci.CursorKind.MACRO_DEFINITION.value,
                                                                    ci.CursorKind.MACRO_INSTANTIATION.value,
                                                                    ci.CursorKind.INCLUSION_DIRECTIVE.value]:
                                    if cn.location.line > pragma_line_num:
                                        if cn.isentity(entity):
                                            cnode = cn
                                        else:
                                            raise RuntimeError(f"error: #pragma ({source.name}:{pragma[0]}) on top of the wrong construct '{cn.entity}', expecting '{entity}'!")
                                        break
                        break

                if cnode is not None:
                    if cnode.pragmas is None:
                        cnode.__attrs__.pragmas = []
                    cnode.__attrs__.pragmas.append(pragma)

    def parse_attributes(self):
        for source in self.sources:

            if source.module is None:
                continue

            attributes = []

            tokens = source.module.__cursor__.get_tokens()

            prev_is_attribute = False
            prev_is_second_open_bracket = False
            prev_is_open_bracket = False
            prev_is_close_bracket = False

            prev_token = None
            attributes = []

            attribute_name = ""
            attribute_dict = {} # {attr1, attr2, ...}
            attribute_elem = [] # attr = (namespace, identifier, arg1, arg2,...)
            args = []
            namespace = ""
            n = 0
            ini_token = None

            for t in tokens:
                is_open_bracket = (t.kind == ci.TokenKind.PUNCTUATION) and (t.spelling == "[")
                is_second_open_bracket = is_open_bracket and prev_is_open_bracket
                is_close_bracket = (t.kind == ci.TokenKind.PUNCTUATION) and (t.spelling == "]")
                is_second_close_bracket = is_close_bracket and prev_is_close_bracket
                is_attribute = prev_is_second_open_bracket or prev_is_attribute

                def gen_attribute(_namespace, _name, _args):
                    if _namespace != "":
                        ns = _namespace + "::"
                    else:
                        ns = ""

                    ret = {f"{ns}{_name}": _args}

                    return ret

                if is_second_open_bracket and ini_token is None:
                    ini_token = prev_token

                if is_attribute:
                    if t.kind == ci.TokenKind.KEYWORD and t.spelling=="using":
                        t = next(tokens)
                        namespace = t.spelling
                        next(tokens)
                    elif t.kind == ci.TokenKind.PUNCTUATION and t.spelling==",":
                        attr = gen_attribute(namespace, attribute_name, args)
                        attribute_dict.update(attr)
                        attribute_name = ""
                        args = []
                    elif t.kind == ci.TokenKind.PUNCTUATION and t.spelling=="(":
                        break_loop = False
                        arg = ""
                        while True:
                            t = next(tokens)
                            if t.kind == ci.TokenKind.PUNCTUATION and t.spelling==",":
                                args.append(arg)
                                arg = ""
                            elif t.kind == ci.TokenKind.PUNCTUATION and t.spelling==")":
                                args.append(arg)
                                arg = ""
                                break_loop = True
                            else:
                                arg += t.spelling

                            if break_loop:
                                break
                    else:
                        attribute_name += t.spelling

                    if n == 0 and is_second_close_bracket:
                        aname = attribute_name[0:-2]
                        if aname in attribute_dict:
                            raise RuntimeError(f"[x] attribute '{aname}' already exists in attribute annotation! ({t.location})")
                        attr = gen_attribute(namespace, aname, args)

                        attribute_dict.update(attr)
                        attributes.append((attribute_dict, ini_token, t))

                        is_attribute = False
                        attribute_name = ""
                        attribute_dict = {}
                        attribute_elem = []
                        namespace = ""
                        args = []
                        n = 0
                        prev_token = None
                        ini_token = None


                    if is_second_open_bracket:
                        n +=1

                    if n > 0 and is_second_close_bracket:
                        n -=1
                else:
                    if len(attributes) > 0:
                        if t.cursor.location == t.location:
                            cnode = self.get_cnode(t.cursor)
                            if cnode.__attrs__.attributes is None:
                                cnode.__attrs__.attributes = []
                            cnode.__attrs__.attributes.extend(attributes.copy())
                            attributes = []

                prev_is_open_bracket = is_open_bracket
                prev_is_second_open_bracket = is_second_open_bracket
                prev_is_close_bracket = is_close_bracket
                prev_is_attribute = is_attribute
                prev_token = t









