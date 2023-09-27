import os.path as osp
from urllib.request import parse_keqv_list
from colorama import Fore, Style
import uuid

import html
import shutil

from heterograph.utils.display import terminal_cpp
from heterograph import hgraph

from . import storage
from .source_manager import CLSourceManager
from .instrumentation import CodeInstrument
from .unparser import pretty_print
from .storage import SourceDiff
from .query import query_visitor
import clang.cindex as ci

'''
AST states:
- aligned: has been formated before cloning. You can only track nodes for aligned asts.
- discardable: by default ASTs are not discardable (cannot be removed),
               cloned ASTs are discardable, and therefore can be removed.

- has_changes: code has been instrumented (annotated), but not yet reflected in the AST
- synced: ast reflects the code (storage)


ast = AST("...") ; ast is not changed + synced
ast.instrument(...) ; ast is changed + synced
ast.commit(...) ; ast is not changed + not sync
ast.sync(...) ; ast is not changed + sync

'''

class Ast(hgraph.HGraph):
    # cmd can be None (get sys.argv) or string

    ### preprocessor: 0: disabled, 1: source (local) #defines, 2: process #if* conditionals, 3: process #includes (global #defines)
    def __init__(self, cmd=None, *, workdir=None, parse=True, preprocessor=None, cxx_spec=None):
        super().__init__()

        # graph/vertex styles
        self.vstyle={'label': Ast.label_vx }
        #self.style={'!on_hover': Ast.on_hover }

        self.cmd = cmd
        self.instr = CodeInstrument()
        self.source_manager = CLSourceManager(ast=self,
                                              cmd=cmd,
                                              workdir=workdir,
                                              cxx_spec=cxx_spec)
        self.read_only = True
        self.is_discardable = False
        self.is_aligned = False

        self.is_parsed = parse

        if self.is_parsed:
            print(f"[i] loading ast {Fore.LIGHTBLACK_EX}(workdir: {self.workdir}){Style.RESET_ALL}")
        else:
            print(f"[i] loading ast [no parsing] {Fore.LIGHTBLACK_EX}(workdir: {self.workdir}){Style.RESET_ALL}")

        if preprocessor is None:
            preprocessor = 1 # preprocess local defs

        self.preprocessor = preprocessor

        if self.is_parsed:
            for src in self.sources:
                self.source_manager.build_module(srcname=src.name, preprocessor_control=self.preprocessor)


    def unmanaged(self):
        self.is_discardable = True


    @property
    def has_changes(self):
        return self.is_parsed and len(self.instr.changed) > 0

    @property
    def workdir(self):
        return self.sources.workdir


    # return ast[cnode-id] => cnode
    def __getitem__(self, idx):
        cnode = self.source_manager.sources.idx_vx.get(idx, None)
        if cnode is None:
            raise RuntimeError("cannot find cnode: %s" % str(idx))
        return cnode

    # finds the closest cnode from (source, line, column)
    def find(self, source, line, column):
        db_elem = self.sources[source]

        tu = db_elem.translation_unit
        file = ci.File.from_name(tu, source)

        # this location is approximate
        location = ci.SourceLocation.from_position(tu, file, line, column)
        cursor = ci.Cursor.from_location(tu, location)

        if cursor is None:
            return None

        cnode = self.source_manager.get_cnode(cursor)

        return cnode

    @property
    def sources(self):
        return self.source_manager.sources

    # nodes0: [ a0, b0, c0, ... ]
    # src0: { srcname: source-code0, ... }
    # src1: { srcname: source-code0, ... }
    # returns: track nodes [ a1, b1, c1, ... ]

    @staticmethod
    def __track(nodes_in, src0, src1, ast1):
        tracked_sources = {}
        for src in src0:
            old_code = src0[src]
            new_code = src1[src]
            tracked_sources[src] = SourceDiff(old_code, new_code)

        lst = []
        for cn in nodes_in:
            loc = cn.location
            lst.append((cn.entity, cn.srcname, loc.line, loc.column))

        ret = []
        for e in lst:
            # e[0] -> kind, e[1] -> source, e[2]-> old line, e[3] -> column
            source = e[1]; column = e[3]
            line = tracked_sources[source].map(e[2])

            if line is None:
                raise RuntimeError("tracked cnode destroyed: %s" % str(e))
            cnode = ast1.find(source, line, column)
            # integerity check
            loc = cnode.location
            if cnode.entity == e[0] and loc.line == line and loc.column == e[3]:
                ret.append(cnode)
            else:
                raise RuntimeError("internal error: tracked cnode integrity is destroyed: %s" % str(e))
        return ret


    def commit(self, *, track=None, preprocessor=None):
        if preprocessor is None:
            preprocessor = self.preprocessor

        if not self.is_parsed:
            raise RuntimeError("cannot commit an unparsed ast!")

        if track is not None:
            if not self.is_aligned:
                raise RuntimeError("tracking nodes disabled: must clone ast with align parameter=True!")

            src0 = { src.name:src.code for src in self.sources }

        print("[i] commiting ast")


        for srcname in self.instr.changed:
           self.source_manager.build_module(srcname=srcname, preprocessor_control=preprocessor)

        self.instr.reset()

        if track is not None:
            src1 = { src.name:src.code for src in self.sources }
            return Ast.__track(track, src0, src1, self)

        return None

    def clone(self, name=None, *, track=None, overwrite=True, changes=None, align=None, parse=True, preprocessor=None, new_args=None, new_defs=None, cxx_spec=None):
        if align is None:
            if not self.is_discardable:
                align = True # self is not a clone, so the clone will be aligned by default
            else:
                align = self.is_aligned #

        if (changes is not None) and not self.is_parsed:
            raise RuntimeError("ast is not parsed: 'changes' parameter cannot be used!")

        if changes is None:
            if self.has_changes:
                raise RuntimeError("ast has changes: use 'changes=True|False' parameter to specify which version to use.")
            changes = False

        if preprocessor is None:
            preprocessor = self.preprocessor

        sources = self.sources

        if name is None:
            while True:
                workdir = osp.abspath(osp.join('/tmp', '.artisan-clones', f'project_{str(uuid.uuid4())[:8]}'))
                # avoid collisions
                if not osp.isdir(workdir):
                    break
        else:
            if osp.isabs(name):
                workdir = name
            else:
                workdir = osp.abspath(osp.join(osp.curdir, name))

        if track is not None:
            if not self.is_aligned:
                raise RuntimeError("cannot track nodes: ast is not aligned!")

            src0 = { src.name:src.code for src in sources }


        if osp.isfile(workdir):
            raise RuntimeError(f"cannot clone: path '{workdir}' is a file!")

        if sources.workdir == workdir:
            raise RuntimeError(f"cannot clone on the same working directory as the original ast: {workdir}!")

        if not overwrite:
            if osp.exists(workdir):
                raise RuntimeError("directory '%s' exists! [overwrite is set to off]" % workdir)

        print(f"[i] cloning: {Fore.BLUE}{self.source_manager.workdir}{Fore.LIGHTBLACK_EX} ▶▶▶ {Fore.LIGHTCYAN_EX}{workdir}{Style.RESET_ALL}")

        storage.copy_tree(sources.workdir, workdir)

        for src in sources:
            filename = osp.join(workdir, src.name)
            if not self.is_parsed:
                with open(filename, 'r') as f:
                    code = f.read()
                    if align:
                        code = pretty_print(code)
                        desc = f"    +- aligning: {Fore.LIGHTCYAN_EX}⥂ {filename}{Style.RESET_ALL}"
                    else:
                        desc = f"    +- writing: {Fore.LIGHTCYAN_EX}{filename}{Style.RESET_ALL}"


            else:
                nchanges = self.instr.num_module_changes(src.translation_unit)
                if not align:
                    desc = f"    +- writing: {Fore.LIGHTCYAN_EX}{filename} ◀◀◀ {Fore.LIGHTBLACK_EX}{nchanges} change(s){Style.RESET_ALL}"
                else:
                    desc = f"    +- writing/aligning: {Fore.LIGHTCYAN_EX}⥂ {filename} ◀◀◀ {Fore.LIGHTBLACK_EX}{nchanges} change(s){Style.RESET_ALL}"

                code = src.module.unparse(changes=changes, align=align)


            with open(filename, 'w+') as f:
                print(desc)
                print(code, file=f)

        if new_args is None:
            _args = self.source_manager.args
        else:
            _args = new_args

        if new_defs is None:
            _defs = self.source_manager.defs
        else:
            _defs = new_defs


        cmd = " ".join([osp.join(workdir, src.name) for src in self.sources]) + " " + \
              " ".join(_args) + " " +\
              " ".join(_defs)

        new_ast = Ast(cmd=cmd, workdir=workdir, parse=parse, preprocessor=preprocessor, cxx_spec=cxx_spec)
        new_ast.is_discardable = True
        if align == True:
            new_ast.is_aligned = True

        if track is not None:
            src1 = { src.name:src.code for src in new_ast.sources }
            ret = Ast.__track(track, src0, src1, new_ast)
            return (new_ast, ret)

        return new_ast


    def sync(self, commit=None):
        if not self.is_parsed:
            raise RuntimeError("cannot sync an unparsed ast!")

        if not self.is_discardable:
            raise RuntimeError("Cannot sync: AST is not discardable (hint: clone first)! ")

        if self.has_changes:
            if commit is None:
                raise RuntimeError("cannot sync: ast contains changes. Hint: use ast.sync(commit=True or False)!")
            if commit:
                self.commit()

        workdir = self.source_manager.workdir
        for src in self.sources:
            filename = osp.join(workdir, src.name)
            with open(filename, 'w+') as f:
                print(f"[i] syncing: {Fore.YELLOW}ast{Fore.LIGHTBLACK_EX} ▶▶▶ {Fore.LIGHTRED_EX}%s{Style.RESET_ALL}" % filename)

                #print("/* MARKED: %s */" % datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'), file=f)
                code = src.module.unparse(changes=True, align=self.is_aligned)
                print(code, file=f)


    def export_to(self, out_dir, *, overwrite=False, changes=None, align=None):
        self.clone(name=out_dir, overwrite=overwrite, changes=changes, parse=False, align=align)

    def unparse(self, **kwargs):
        code = ""
        for source in self.sources:
            if len(self.sources) > 1:
                code += f"===========[{source.name}]============\n"
            else:
                code += "\n"
            code += source.module.unparse(**kwargs)

        return code

    def exec(self, rule='', *, makefile="meta_cl.make", force=False, env=None, report=None, addr="*", port=9865):
        if not force:
            if self.has_changes:
                raise RuntimeError("cannot execute: ast has changes! (hint: use force=True ignore changes)")

        # env {'VAR0', 'val0', 'VAR1': 'val1'}
        import subprocess
        import os
        import threading
        from queue import Queue

        print(f"[i] executing: {Fore.YELLOW}ast{Fore.LIGHTBLACK_EX} ({Fore.LIGHTBLACK_EX}{makefile}){Style.RESET_ALL}")

        if (not osp.exists(osp.join(self.source_manager.workdir, makefile))):
            raise RuntimeError("expecting '{}' in base directory '{}' to execute ast!".format(makefile, self.sources.workdir))

        # build and execute
        cmd = ['make', '-f', makefile]
        if rule != '':
            cmd.append(rule)

        _env = os.environ.copy()
        if env is not None:
           _env.update(env)

        # extra ast vars
        _env.update({'AST_WORKDIR': self.source_manager.workdir,
                     'AST_SOURCES': " ".join([osp.join(self.source_manager.workdir, s.name) for s in self.sources]),
                     'AST_DEFS': " ".join(self.source_manager.defs),
                     'AST_ARGS': " ".join(self.source_manager.args)
                     })

        def thread_run(cmd, env, que):
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.sources.workdir)
            while True:
                output = proc.stdout.readline()
                if output == b'' and proc.poll() is not None:
                    break
                if output:
                    print(f"{Fore.LIGHTCYAN_EX}> %s{Style.RESET_ALL}" % output.strip().decode('utf8'))
            rc = proc.poll()
            que.put(rc)

        que = Queue()
        thread = threading.Thread(target=thread_run, args=(cmd, _env, que))

        if report is not None:
            # fn => def report(ast, data)
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.PULL)
            socket.bind('tcp://{}:{}'.format(addr, port))

            thread.start()

            while thread.is_alive():
                result = socket.recv().decode('utf-8')
                if result != "":
                    report(self, eval((result)))
                else:
                    break

        else:
            thread.start()
            thread.join()

        return que.get()


    def reset(self):
        if self.has_changes:
            self.instr.reset()

    def discard(self):
        if not self.is_discardable:
            raise RuntimeError("AST is non-discardable! (hint: clone it)")

        shutil.rmtree(self.source_manager.workdir)
        print(f"[i] ast discarded: {Fore.LIGHTRED_EX}{self.source_manager.workdir}{Style.RESET_ALL}")
        del self

    def query(self, select, **kwargs):
        # self.source is a property of HGRaph, which corresponds to the module
        return query_visitor(self, vs=self.source, select=select, **kwargs)

    def tree(self, info=None, depth=-1):
        for source in self.sources:
            if source.module:
                source.module.tree(info, depth)

    def parse_pragmas(self, rules=None):
        self.source_manager.parse_pragmas(rules)

    def parse_attributes(self):
        self.source_manager.parse_attributes()

    ############################# graph viewing ###
    @staticmethod
    def label_vx(ast, vx):
        cnode = ast[vx]
        header = cnode.entity
        extent = cnode.__cursor__.extent

        loc = html.escape("(%d:%d)-(%d:%d)" % (extent.start.line,
                                               extent.start.column,
                                               extent.end.line,
                                               extent.end.column))

        spec = r'<TABLE BORDER="0"><TR><TD ALIGN="CENTER"><B>%s (%d)</B></TD></TR><TR><TD ALIGN="CENTER"><b><FONT COLOR="#BA2E00"> %s</FONT></b></TD></TR> </TABLE>' % (header, vx, loc)
        return r'<%s>' % spec

    @staticmethod
    def on_hover(ast, elem):
        if type(elem) == int:
            cnode = ast[elem]
            code = cnode.unparse(align=True)

            print ("========================[cnode: %d]===" % cnode.id)
            print(terminal_cpp(code))
            return True
        return False







