#!/usr/bin/env artisan

from meta_cl import *
from util import * 

# def omp_loop_pragmas(pragma):
#     if " ".join(pragma[0:3]) == "omp parallel for":
#         return "loop" 
#     else:
#         return False

# def omp_loop_filter(loop):
#     if loop.pragmas is not None:
#         print(loop.pragmas)
#         return True
#     return False

class LoopTimeProfiler:
    def __init__(self, ast):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-loop-profiler/', changes=True) 
        self.data = {}

    # returns profiling data
    def run(self, filter_fn=outermost_filter, debug=False, exec_rule=''):

        self.ast.parse_pragmas(rules=[parse_omp_parallel_loop_pragmas])
        # finds all for-loops that match a specified filter (default: outermost loops) 
        if not filter_fn:
            filter_fn = lambda l: True
        results = self.ast.query(select="src{Module} => loop{ForStmt}",
                                 where=lambda loop: filter_fn(loop))
        if len(results) == 0:
            return 0

        # include header <meta_cl> on all matched sources
        srcs = results.apply(RSet.distinct, target='src')
        for res in srcs:
            incl_artisan(res.src)

        # instrument matched loops to add a timer
        for res in results:
            if res.loop.pragmas is not None:
                pragmas = ""
                for p in res.loop.pragmas:
                    pragmas += "#pragma " + " ".join(p[0]) + "\n"
                res.loop.instrument(Action.pragmas, fn=lambda p: False)
                res.loop.instrument(Action.replace, code='{ Timer __timer_%s__([](double t) { Report::write("(\'%s\',%%1%%),", t); }, true);\n%s%s}\n' % (res.loop.tag, res.loop.tag, pragmas, res.loop.unparse()))
            else:
                res.loop.instrument(Action.before, code='{ Timer __timer_%s__([](double t) { Report::write("(\'%s\',%%1%%),", t); }, true);\n' % (res.loop.tag, res.loop.tag))
                res.loop.instrument(Action.after, code='}')
            
        # query main function
        main_fn = self.ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
        if not main_fn:
            return 
        main_fn = main_fn[0].fn

        # instrument main function with a timer
        main_fn.instrument(Action.begin, code='Report::connect(); Report::write("["); int ret;'
                                        '{ Timer __timer_main__([](double t) { Report::write("(\'main_fn\',%1%),", t); }, true);'
                                        'ret = [] (auto argc, auto argv) { ')
        main_fn.instrument(Action.end, code='}(argc, argv);\n}'
                                        'Report::write("]");\nReport::emit();\nReport::disconnect();'
                                        'return ret;')

        # sync AST to execute
        self.ast.sync(commit=True)

        # run
        def report(ast, data):
            for (tag, time) in data:
                if tag not in self.data:
                    self.data[tag] = 0
                self.data[tag] += time
        self.ast.exec(report=report, rule=exec_rule)

        if not debug:
            self.ast.discard()

class LoopTripCountProfiler:
    # TODO: ideally don't limit to a fn -- use node tracing across clones to traces scope 
    def __init__(self, ast, fn_name):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-loop-tripcount-profiler/', changes=True) 
        self.fn = self.ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)[0].fn
        self.data = {}

    # returns profiling data
    def run(self, filter_fn=None, debug=False, exec_rule=''):
        
        # finds all loops that match an optional filter 
        if not filter_fn:
            filter_fn = lambda l: True
        results = self.fn.query(select="(floop{ForStmt}|wloop{WhileStmt})",
                                 where=lambda floop, wloop: floop and filter_fn(floop) or wloop and filter_fn(wloop))
        if len(results) == 0:
            return 0

        # instrument matched loops to count iterations
        count_inits = ""
        count_reports = ""
        srcs = []
        outermost_loop = None
        for res in results:
            if res.floop:
                loop = res.floop 
            elif res.wloop:
                loop = res.wloop 
            else:
                break
            if outermost_loop == None or loop.depth <= outermost_loop.depth:
                outermost_loop = loop
            srcs.append(loop.module)
            count_inits += 'unsigned long total_count_%s = 0;\n unsigned long invocation_count_%s = 0;\n' % (loop.tag, loop.tag)
            
            loop.instrument(Action.begin, code='total_count_%s++;\n' % (loop.tag))
            loop.instrument(Action.before, code='invocation_count_%s++;\n' % (loop.tag))
            count_reports += 'Report::write("(\'%s_total\',%%1%%),", total_count_%s);\n' % (loop.tag, loop.tag)
            count_reports += 'Report::write("(\'%s_invoc\',%%1%%),", invocation_count_%s);\n' % (loop.tag, loop.tag)

        # include header <meta_cl> on all matched sources
        srcs = list(set(srcs))
        for src in srcs:
            # src.instrument(Action.before, code="#include <meta_cl>\nusing namespace meta_cl;\n")
            incl_artisan(src)

        # instrument function to initialise loop trip counts at the beginning and report values at the end
        report_string = 'Report::connect();\n Report::write("[");\n' + count_reports + 'Report::write("]");\nReport::emit();\nReport::disconnect();'
        fn_type = self.fn.signature.rettype.spelling
        if fn_type == 'void':
            self.fn.instrument(Action.begin, code=count_inits)
            self.fn.instrument(Action.end, code=report_string)
        else:
            # handle non-void functions with a wrapper to allow return
            sig = ', '.join(["%s %s" % (p[0].spelling, p[1]) for p in self.fn.signature.params])
            param_names = ', '.join([p[1] for p in self.fn.signature.params])
            ## HACK: report counts after outermost loop to avoid instrumenting after return
            outermost_loop.instrument(Action.after, code=count_reports)
            self.fn.instrument(Action.begin, code = '%s ret; Report::connect(); Report::write("["); \n ret = [=] (%s) {\n  %s ' % (fn_type, sig, count_inits))
            self.fn.instrument(Action.end, code = '}(%s);\n Report::write("]");\nReport::emit();\nReport::disconnect();return ret;' % (param_names))

        # sync AST to execute
        self.ast.sync(commit=True)
        
        # run
        def report(ast, data):
            for (tag, count) in data:
                looptag = tag[:-6]
                if not looptag in self.data:
                    self.data[looptag] = {}
                if '_total' in tag:
                    self.data[looptag]['total'] = count
                elif '_invoc' in tag:
                    self.data[looptag]['instances'] = count

        self.ast.exec(report=report, rule=exec_rule)

        if not debug:
            self.ast.discard()

        return self.data


class PointerRangeProfiler:
    def __init__(self, ast, fn_name):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-pointer-analysis/', changes=True)
        self.scope = self.ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)[0].fn
        self.data = {}

    def run(self, debug=False, exec_rule='small'):

        # include header <meta_cl> on all matched sources
        srcs = self.ast.query(select="src{Module}")
        for res in srcs:
            incl_artisan(res.src)

        # instrument main function to connect and disconnect from report pipe
        main_fn = self.ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
        if not main_fn:
            return
        main_fn = main_fn[0].fn
        main_fn.instrument(Action.begin, code='Report::connect(); Report::write("["); int ret; ret = [](auto argc, auto argv) {')
        main_fn.instrument(Action.end, code='}(argc, argv); Report::disconnect(); return ret;')

        # find all pointer dereferences in this scope and any called functions
        derefs = [row.ase for row in self.scope.query('ase{ArraySubscriptExpr}')]
        called_fns = get_called_fns(self.ast,self.scope)
        for fn_ in called_fns:
            derefs += [row.ase for row in fn_[1].query('ase{ArraySubscriptExpr}')]
        
        # instrument code to report address of all pointer dereferences
        nodes_to_instrument = {}
        scopes_to_instrument = []
        loops_to_instrument = {}
        for deref in derefs:
            # construct report string 
            arr = deref.children[0].unparse()
            idx = deref.children[1].unparse()
            report_str = 'Report::write("(\'%s\', %%1%%),", (unsigned long int) %s);' % (arr, arr+"+("+idx+")")
            # find where to instrument
            node = deref
            while node and not node.parent.isentity('CompoundStmt') and not node.parent.isentity('ForStmt') and not node.parent.isentity('WhileStmt'):
                node = node.parent
            if not node:
                print("Something went wrong 1.")
                exit(1)
            
            if node.parent.isentity('CompoundStmt'):
                # track nodes which need instrumentation (may be multiple dereferences per node)
                if not node.id in nodes_to_instrument:
                    nodes_to_instrument[node.id] = []
                nodes_to_instrument[node.id].append(report_str)
                # track scopes to instrument with emit
                if not node.parent in scopes_to_instrument:
                    scopes_to_instrument.append(node.parent)

            # handle cases where dereference is in a loop condition
            elif node.parent.isentity('ForStmt') or node.parent.isentity('WhileStmt'):
                loop = node.parent
                if loop.id not in loops_to_instrument:
                    loops_to_instrument[loop.id] = []
                loops_to_instrument[loop.id].append(report_str)

                # find next outer scope to instrument with emit 
                parent = loop.parent
                while parent and not parent.isentity('CompoundStmt'):
                    parent = parent.parent
                if not parent:
                    print("Something went wrong 2.")
                    exit(1)
                if not parent in scopes_to_instrument:
                    scopes_to_instrument.append(parent)
            else:
                print("Something went wrong 3.")
                exit(1) 

        # instrument any scope thats not enclosed in another scope with an emit 
        for scope in scopes_to_instrument:
            enclosed_in_another_scope = any([other_scope.encloses(scope) for other_scope in [s for s in scopes_to_instrument if s != scope]])
            if not enclosed_in_another_scope:
                scope.instrument(Action.end, code='Report::emit("]");Report::write("[");')

        # separate loop performs instrumentation to avoid instrumenting the same node multiple times
        for node in nodes_to_instrument:
            # instument *before* to avoid cases where idx changes inside stmt (e.g. conditional)
            self.ast[node].instrument(Action.before, code="%s" % '\n'.join(nodes_to_instrument[node]))
        
        # instrument loops differently (need before, and at begin of body)
        for loop in loops_to_instrument:
            self.ast[loop].instrument(Action.before, code="%s" % '\n'.join(loops_to_instrument[loop]))
            self.ast[loop].body.instrument(Action.begin, code="%s" % '\n'.join(loops_to_instrument[loop]))

        self.ast.sync(commit=True)
        
        def report(ast, data):
            for arr, addr in data:
                if arr not in self.data:
                    self.data[arr] = {'min': addr, 'max': addr}
                else:
                    self.data[arr]['min'] = min(addr, self.data[arr]['min'])
                    self.data[arr]['max'] = max(addr, self.data[arr]['max'])
        
        self.ast.exec(report=report, rule=exec_rule)
        
        if not debug:
            self.ast.discard()


class DataInOutProfiler:
    def __init__(self, ast):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-datainout-profiler/', changes=True) 
        self.data = {}

    def run(self, fn_name, exec_rule='', debug=False):

        # find top level source with main function 
        results = self.ast.query(select="src{Module} => fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
        if not results:
            return
        main_src = results[0].src

        # instrument with include and ARTISAN MEM INIT
        # main_src.instrument(Action.before, code="#include <meta_cl>\nusing namespace meta_cl;\nARTISAN_MEM_INIT\n")
        incl_artisan(main_src, mem_init=True)

        other_srcs = self.ast.query(select="src{Module}", where=lambda src: src.tag != main_src.tag)
        for res in other_srcs:
            # res.src.instrument(Action.before, code="#include <meta_cl>\nusing namespace meta_cl;\n")
            incl_artisan(res.src)

        # find all dynamic pointer allocations in application, instrument to register
        register_dynamic_pointer_sizes(self.ast)

        # instrument static pointer allocations to register size 
        register_static_array_sizes(self.ast) 

        # for given function, find pointer arguments 
        results = self.ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == fn_name)
        if not results:
            return
        fn = results[0].fn
        pointer_params = fn.query('p{ParmDecl}', where=lambda p: pointer_type(p.type) or array_type(p.type))

        # instrument function to report pointer sizes and determine which pointer
        # args are read (input) and write (output)
        reporting_string = 'Report::connect(); Report::write("[");\n'
        for row in pointer_params:
            reporting_string += 'Report::write("(\'%s\',%%1%%),", Mem::size(%s));\n' % (row.p.name, row.p.name)
            rw = read_or_write_param(row.p, fn)
            self.data[row.p.name] = {'rw': rw, 'type': row.p.type.spelling, 'size': None}
        
        reporting_string += 'Report::emit("]"); Report::disconnect();'
        fn.instrument(Action.begin, code=reporting_string)

        # sync AST to execute
        self.ast.sync(commit=True)

        # run
        def report(ast, data):
            self.data['summary'] = {'bytes_in': 0,'bytes_out': 0}
            for (ptr, size) in data:
                self.data[ptr]['size'] = size
                if 'R' in self.data[ptr]['rw']:
                    self.data['summary']['bytes_in'] += size
                if 'W' in self.data[ptr]['rw']:
                    self.data['summary']['bytes_out'] += size
            self.data['summary']['bytes_inout'] = self.data['summary']['bytes_in']+self.data['summary']['bytes_out']

        self.ast.exec(report=report, rule=exec_rule)

        if not debug:
            self.ast.discard()


class MemoryFootprintProfiler:
    def __init__(self, ast):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('memory-footprint-profiler/', changes=True) 
        self.data = {}

    def run(self, fn_name, exec_rule='', debug=False):

        # find top level source with main function, instrument to include artisan
        results = self.ast.query(select="src{Module} => fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
        if not results:
            print("Can't find top level source, exiting...")
            return
        main_src = results[0].src
        incl_artisan(main_src)

        # find all array/pointer variable references in the specified function / scope
        results = self.ast.query("fn{FunctionDecl}=>ase{ArraySubscriptExpr}=>ref{DeclRefExpr}", where=lambda fn: fn.name == fn_name)
        if not results:
            print("No memory accesses in specified scope, exiting...")
            exit()
        fn = results[0].fn

        # count memory accesses (B read / write)
        to_instrument = {}
        pointers = []
        for row in results:
            
            if not (pointer_type(row.ref.type) or array_type(row.ref.type)):
                continue
            
            # extract memory access information
            var = row.ref.name
            idx = row.ase.children[1].unparse()
            rw = read_or_write(row.ref) 
            # figure out access type (find parent memberref if needed)
            # i.e. access p[i].x should have type of x, not struct type of p
            access = row.ref
            while access.parent and access.parent.isentity('ArraySubscriptExpr') or access.parent.isentity('MemberRefExpr'):
                access = access.parent
            access_type = access.type.spelling

            # track all pointers to count 
            pointers.append(var)

            # find statement to instrument after
            # TODO: doesn't handle derefs in for/while conditions 
            stmt = row.ref   
            while stmt and not stmt.parent.isentity('CompoundStmt') and not stmt.parent.isentity('ForStmt') and not stmt.parent.isentity('WhileStmt'):
                stmt = stmt.parent
            if not stmt:
                print("Can't find stmt to instrument, exiting...")
                exit()

            # derive byte count update strings using conditional stmts to
            # avoid double counting subsequent accesses to the same idx
            count_updates = ""
            if 'R' in rw:
                # count_updates += f"if ( ({idx}) != {var}_read_idx_) {{ {var}_read_count__+=sizeof({access_type}); {var}_read_idx_ = ({idx}); }}" 
                count_updates += f"{var}_read_count__+=sizeof({access_type});" 
            if 'W' in rw:
                # count_updates += f"if ( ({idx}) != {var}_write_idx_) {{ {var}_write_count__+=sizeof({access_type}); {var}_write_idx_ = ({idx}); }}"
                count_updates += f"{var}_write_count__+=sizeof({access_type});" 
 
            if stmt not in to_instrument:
                to_instrument[stmt] = []
            to_instrument[stmt] += [count_updates]

        # instrument statements with byte count update strings
        for node in to_instrument:
            node.instrument(Action.after, code=';\n'+'\n'.join(to_instrument[node]))

        # initialise and report all access byte counts
        pointers = set(pointers)
        init_counts = []
        report_counts = []
        for var in pointers:
            init_counts.append(f"long {var}_write_count__ = 0; int {var}_write_idx_ = -1;")
            init_counts.append(f"long {var}_read_count__ = 0; int {var}_read_idx_ = -1;")
            report_counts.append(f"Report::write(\"(\'{var}_read\',%1%),\", {var}_read_count__);")
            report_counts.append(f"Report::write(\"(\'{var}_write\',%1%),\", {var}_write_count__);")

        # initialise byte counts at the beginning of function
        fn.instrument(Action.begin, code='\n'.join(init_counts))

        # report byte counts at the end of function
        fn.instrument(Action.end, code='Report::connect();Report::write("[");\n%s\nReport::emit("]");Report::disconnect();' % '\n'.join(report_counts))

        # sync AST to execute
        self.ast.sync(commit=True)

        # run
        def report(ast, data):
            total_R = 0
            total_W = 0
            data = dict((x, y) for x, y in data)
            for pointer in pointers:
                self.data[pointer] ={'bytes_R': data[pointer+'_read'], 'bytes_W': data[pointer+'_write']}
                total_R += data[pointer+'_read']
                total_W += data[pointer+'_write']
            self.data['__TOTAL__'] = {'bytes_R': total_R, 'bytes_W': total_W}

        self.ast.exec(report=report, rule=exec_rule)

        if not debug:
            self.ast.discard()


class HIPKernelTimer:
    def __init__(self, ast):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-kernel-timer/', changes=True) 
        self.data = None

    # returns profiling data
    def run(self, num_runs=1, debug=False):

        # find kernel launch function call - hip macro instantiation
        kernel_launch = self.ast.query('src{Module} => mi{MacroInstantiation}', where=lambda mi: mi.unparse().startswith("hipLaunchKernelGGL"))
        if len(kernel_launch) == 0:
            print("Can't find kernel launch function to time. Returning.")
            return
        src1 = kernel_launch[0].src
        kernel_launch = kernel_launch[0].mi
        
        # find kernel wrapper function call 
        kernel_wrapper = self.ast.query(select='src{Module} => call{CallExpr}', where=lambda call: call.name == '__kernel___wrapper_')
        if len(kernel_wrapper) == 0:
            print("Can't find kernel wrapper function to time. Returning.")
            return

        src2 = kernel_wrapper[0].src
        kernel_call = kernel_wrapper[0].call
        srcs = set([src1, src2])
        
        # include header <artisan> on sources
        for src in srcs:
            src.instrument(Action.before, code="#include <artisan>\n")

        # instrument wrapper function to add a timer
        kernel_call.instrument(Action.before, code='{ artisan::Timer __kernel_timer__([](double t) { artisan::Report::write("- e2e-timer: %f", t); });')
        kernel_call.instrument(Action.after, code=';}')

        # instrument launch call to add a timer and synchronise mechanisms 
        kernel_launch.instrument(Action.before, code='hipDeviceSynchronize();\n{ artisan::Timer __kernel_timer__([](double t) { artisan::Report::write("- compute-timer: %f", t); });\n')
        kernel_launch.instrument(Action.after, code=';\nhipDeviceSynchronize();\n}')

        # step 3: query main function
        results2 = self.ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == 'main')

        for res in results2:
            res.fn.instrument(Action.begin,
                              code='artisan::Report::start(); int ret = [] (auto argc, auto argv) { ')
            res.fn.instrument(Action.end,
                              code='  }(argc, argv);'
                                   'artisan::Report::emit("profile");'
                                   'artisan::Report::end();'
                                   'return ret;')

        # sync AST to execute
        self.ast.sync()

        # run
        def report(ast, data):
            self.data = data

        e2e_times = []
        compute_times = []
        for iter in range(0,num_runs):
            self.ast.exec(rule='gpu')
            self.ast.exec(rule='run_gpu', report=report)
            compute_time = 0
            e2e_time = 0 
            for el in self.data['profile']:
                if 'compute-timer' in el:
                    compute_time += el['compute-timer']
                if 'e2e-timer' in el:
                    e2e_time += el['e2e-timer']
            compute_times.append(compute_time)
            e2e_times.append(e2e_time)

        if not debug:
            subprocess.run(['cat', self.ast.workdir+"/outputs.txt"])
            self.ast.discard()

        return e2e_times, compute_times