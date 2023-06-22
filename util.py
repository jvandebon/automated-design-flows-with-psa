#!/usr/bin/env artisan
from meta_cl import *
import json
import islpy as isl
import random

def incl_artisan(src, mem_init=False):
    includes_artisan = src.query('inc{InclusionDirective}', where=lambda inc: inc.unparse() == "#include <meta_cl>")
    if not mem_init and not includes_artisan:
        src.instrument(Action.before, code="#include <meta_cl>\nusing namespace meta_cl;\n")
    elif mem_init:
        mem_init_macro = src.query('macro{MacroInstantiation}', where=lambda macro: macro.unparse() == "META_CL_MEM_INIT")
        if not includes_artisan and not mem_init_macro:
            src.instrument(Action.before, code="#include <meta_cl>\nusing namespace meta_cl;\nMETA_CL_MEM_INIT\n")
        elif includes_artisan and not mem_init_macro:
            includes_artisan[0].inc.instrument(Action.after, code="META_CL_MEM_INIT\n")

def outermost_filter(loop):
    return loop.is_outermost()

def locations_filter(loop, locations):
    return str(loop.location) in locations

def parallel_filter(loop):
    reads, writes, schedule = build_polyhedral_model(loop)
    parallel = is_parallel(reads,writes,schedule,get_loop_info(loop)['idx'])
    return parallel

def parse_omp_parallel_loop_pragmas(pragma):
    if " ".join(pragma[0:3]) == "omp parallel for":
        return "loop" 
    else:
        return False # exclude pragma

def omp_parallel_loop_filter(loop):
    return loop.pragmas is not None and any([p[0] for p in loop.pragmas if len(p[0]) > 3 and " ".join(p[0][0:3]) == "omp parallel for"])

def get_all_referenced_vars(node):
    decl_refs = node.query("dr{DeclRefExpr}", where=lambda dr: not function_type(dr.type))
    return [row.dr for row in decl_refs]

def get_global_var_refs(refs):
    # assumes anything referenced and not defined in this scope is global 
    return [ref.decl for ref in refs if ref.decl and not ref.decl.is_local]

def get_local_var_list(node):
    var_decls = node.query("ds{DeclStmt}=>vd{VarDecl}")
    return [row.vd for row in var_decls]

# TODO: will not work with overloaded functions (modify to check signature)
def get_called_fns_(ast, fn, results):
    calls = fn.query('c{CallExpr}')
    for row in calls:
        call_name = row.c.name
        fn_called = ast.query('fd{FunctionDecl}', where=lambda fd : fd.name == call_name)
        if not fn_called or len(fn_called) > 1:
            continue
        if (call_name, fn_called[0].fd) not in results:
            results.append((call_name, fn_called[0].fd))
            get_called_fns_(ast,fn_called[0].fd,results)

def get_called_fns(ast, fn):
    results = []
    get_called_fns_(ast,fn,results)
    return results

def range_overlaps(range1, range2):
    if range1['min'] <= range2['max'] and range1['max'] >= range2['min']:
        return True
    return False

def build_points_to_map(pointer_ranges):
    # determine if any pointers overlap, build points-to graph
    alias_map = {}
    for p1 in pointer_ranges:
        alias_map[p1] = []
        for p2 in [p for p in pointer_ranges if p != p1]:
            if range_overlaps(pointer_ranges[p1], pointer_ranges[p2]):
                alias_map[p1].append(p2)
    return alias_map

def is_reachable(n1, n2, adj_map):
    # bfs approach to check if there is a path between two nodes
    visited = []
    q = []
    visited.append(n1) 
    q.append(n1)
    while (len(q) > 0):
        n = q.pop(0)
        if n not in adj_map:
            continue
        for i in adj_map[n]:
            if (i == n2):
                return True
            if not i in visited:
                visited.append(i)
                q.append(i)
    return False

def function_type(t):
    return str(t.kind) == "TypeKind.FUNCTIONPROTO"
    
def pointer_type(t):
    return str(t.kind) == "TypeKind.POINTER"

def array_type(t):
    return str(t.kind) == "TypeKind.CONSTANTARRAY"   

def read_or_write_param(param, fn): # determine if a fn param is read or write or both 
    refs = fn.query("ref{DeclRefExpr}", where=lambda ref: ref.name == param.name)
    rw =  ""
    if any([row.ref.access.is_used for row in refs]):
        rw += 'R'
    if any([row.ref.access.is_def for row in refs]):
        rw += 'W'
    return rw

def read_or_write(ref): # determine if a variable reference is reading or writing or both
    rw = ""
    if ref.access.is_used:
        rw += 'R'
    if ref.access.is_def:
        rw += 'W'
    return rw

def inline_fn(ast, scope, call, fn=None):
    if not fn:
        fn = ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == call.name and fn.body)
    if not fn: # function not locally defined, cannot inline
        return 
    fn = fn[0].fn
    fn_params, call_args = fn.signature.params, call.args
    inlined_code = ""
    ret = fn.query("r{ReturnStmt}")
    # initialise func params as call args
    # note: assumes no pointer args are derived (e.g. f(a+10) or f(&a[10]) where a is a pointer)
    for i in range(len(fn_params)):
        param_type,  param_name = fn_params[i][0], fn_params[i][1] 
        arg = call_args[i].unparse()
        if arg != param_name:
            cast = ""
            if pointer_type(param_type):
                cast = "(%s)"%param_type.spelling
            inlined_code += "%s %s = %s%s;\n" % (param_type.spelling,param_name,cast,arg)
    # insert func body 
    inlined_code += fn.body.unparse()[1:-1]
    # instrument to replace call with inlined  fn 
    if ret:
        ret_stmt = ret[0].r.unparse().strip()
        inlined_code = inlined_code.replace(ret_stmt, "") # remove return stmt from inline
        ret_value = ret[0].r.children[0].unparse()
        call.parent.instrument(Action.before, code=inlined_code) ## TODO: assumes binaryoperation (x=f())
        call.instrument(Action.replace, code=ret_value)
    else:
        call.instrument(Action.replace, code=inlined_code)
        call.instrument(Action.remove_semicolon, verify=False)

def pragma_to_dict(pragma):
    if not pragma:
        return {}
    return json.loads(''.join(pragma[0][3:]))

# TODO: make robust to symbols 
def try_eval(expr):
    try:
        return eval(expr)
    except:
        return expr

# TODO: currently only supports (T i = N; i < M; i++) kind of loops
def get_loop_info(loop):
    vd = loop.init.query("v{VarDecl}")
    if not vd or len(vd) > 1:
        print("Not a well-formed loop: %s" % loop.init.unparse(), loop.condition.unparse())
        return None
    idx = vd[0].v.name
    start = vd[0].v.children[-1].unparse()
    if not loop.condition.isentity("BinaryOperator") and not len(loop.condition.children) == 2:
        print("Not a well-formed loop: %s" % loop.init.unparse(), loop.condition.unparse())
        return None
    end = loop.condition.children[1].unparse()
    return {'idx': idx, 'start': str(try_eval(start)), 'end': str(try_eval(end))}


def register_dynamic_pointer_sizes(ast, scopes_to_ignore=[]):
    # TODO: currently no support for 'realloc'
    # allocations with malloc
    results = ast.query('bop{BinaryOperator} => c{CallExpr}', where=lambda c: c.name == 'malloc')
    for row in results:
        malloc_call = row.c
        decl_ref = row.bop.children[0]
        if not decl_ref.isentity('DeclRefExpr'):
            continue
        row.bop.instrument(Action.after, code=";\nMem::reg(%s, %s);\n" % (decl_ref.name, malloc_call.args[0].unparse()))
        row.bop.instrument(Action.remove_semicolon, verify=False)
    # allocations with calloc
    results = ast.query('bop{BinaryOperator} => c{CallExpr}', where=lambda c: c.name == 'calloc')
    results = [row for row in results if not any([scope.encloses(row.c) for scope in scopes_to_ignore])]
    for row in results:
        calloc_call = row.c
        decl_ref = row.bop.children[0]
        if not decl_ref.isentity('DeclRefExpr'):
            continue
        # TODO: calloc_call.args does not catch first arg
        args = calloc_call.unparse().replace("calloc(", "")[:-1].split(",")
        size_str = args[0] + "*"+ args[1]
        row.bop.instrument(Action.after, code=";\nMem::reg(%s, %s);\n" % (decl_ref.name, size_str))
        row.bop.instrument(Action.remove_semicolon, verify=False)
    # allocations with 'new'
    results = ast.query('new{CxxNewExpr}')
    results = [row for row in results if not any([scope.encloses(row.new) for scope in scopes_to_ignore])]
    for row in results:
        code = row.new.unparse()
        num_els = code[code.index('[')+1:code.index(']')]
        ptype = row.new.type.spelling.replace("*", "").strip()
        size_str = "%s*sizeof(%s)" % (num_els, ptype)
        if row.new.parent.isentity('VarDecl'): # T * a = new ...
            name = row.new.parent.name
        elif row.new.parent.isentity('BinaryOperator'): # a = new ... 
            name = row.new.parent.children[0].name
        else:
            # TODO: are there any other cases to cover?
            continue
        row.new.parent.instrument(Action.after, code=";\nMem::reg(%s, %s);\n" % (name, size_str))
        row.new.parent.instrument(Action.remove_semicolon, verify=False)

def register_static_array_sizes(ast, scopes_to_ignore=[]):
    results = ast.query("decl{DeclStmt}=>vd{VarDecl}", where=lambda vd:array_type(vd.type))
    results = [row for row in results if not any([scope.encloses(row.decl) for scope in scopes_to_ignore])]
    for row in results:
        row.decl.instrument(Action.after, code="Mem::reg(%s, sizeof(%s));\n" % (row.vd.name, row.vd.name))
        row.decl.instrument(Action.remove_semicolon, verify=False)

def trace_memory(ast, scopes_to_ignore=[]):
    # find top level source with main function 
    results = ast.query(select="src{Module} => fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
    if not results:
        return
    main_src = results[0].src
    # instrument with include and ARTISAN MEM INIT
    incl_artisan(main_src, mem_init=True)
    other_srcs = ast.query(select="src{Module}", where=lambda src: src.tag != main_src.tag)
    for res in other_srcs:
        incl_artisan(res.src)
    # find all dynamic pointer allocations in application, instrument to register
    register_dynamic_pointer_sizes(ast, scopes_to_ignore=scopes_to_ignore)
    # instrument static pointer allocations to register size 
    register_static_array_sizes(ast, scopes_to_ignore=scopes_to_ignore) 

# Polyhedral Analysis Helpers

# TODO: check more cases, see if this generalisation can work 
def dep_map_handler(m, deps):
    d = m.domain()
    r = m.range()
    sink = r.get_tuple_name()
    exp = m.simple_hull()
    constraints = [c for c in exp.get_constraints() if c.is_equality()]
    dep = {'sink': sink, 'src': d.get_tuple_name(), 'dists':[]}
    for c in constraints:
        coefs = c.get_coefficients_by_name()
        var = ""
        dist = 0
        for i in coefs:
            if i == 1:
                dist = int(str(coefs[i]))
            else:
                var = i
        dep['dists'].append((var, dist))
    if not sink in deps:
        deps[sink] = []
    deps[sink].append(dep)

#  TODO: if there are multiple deps in one stmt, match var to dep
def sink_map_handler(m, vars):
    stmt = m.domain().get_tuple_name()
    var = m.range().get_tuple_name()
    if not stmt in vars:
        vars[stmt] = []
    vars[stmt].append(var)

def handle_deps(flow, sinks):
    deps = flow[1]
    no_source = flow[3]
    dependent_sinks = sinks.subtract(no_source)
    dep_vars = {}
    dependent_sinks.foreach_map(lambda m: sink_map_handler(m, dep_vars))
    dep_details = {}
    deps.foreach_map(lambda m: dep_map_handler(m, dep_details))
    return dep_details, dep_vars

def analyse_raw_deps(reads, writes, schedule):
    flow = reads.compute_flow(writes, writes, schedule)
    deps = flow[1] 
    if deps.is_empty():
        return False
    return handle_deps(flow, reads)

def analyse_war_deps(reads, writes, schedule):
    flow = writes.compute_flow(reads, reads, schedule)
    deps = flow[1] 
    if deps.is_empty():
        return False
    return handle_deps(flow, writes)

def analyse_waw_deps(reads, writes, schedule):
    flow = writes.compute_flow(writes, writes, schedule)
    deps = flow[1] 
    if deps.is_empty():
        return False
    return handle_deps(flow, writes)

def parallel_deps(dep_details, idx_var):
    parallel = True
    for stmt in dep_details:
        for dep in dep_details[stmt]:
            for dist in [d[1] for d in dep['dists'] if d[0] == idx_var]:
                if dist != 0:
                    parallel = False
    return parallel

def is_loop_parallel(raw_deps, war_deps, waw_deps, idx_var):
    if not raw_deps and not war_deps and not waw_deps:
        return True
    if raw_deps and not parallel_deps(raw_deps[0], idx_var):
        return False
    if war_deps and not parallel_deps(war_deps[0], idx_var):
        return False 
    if waw_deps and not parallel_deps(waw_deps[0], idx_var):
        return False 
    return True

def analyse_loop_deps(reads, writes, schedule, debug=False):
    raw_deps = analyse_raw_deps(reads, writes, schedule)
    war_deps = analyse_war_deps(reads, writes, schedule)
    waw_deps = analyse_war_deps(reads, writes, schedule)
    return raw_deps, war_deps, waw_deps

def process_stmts(loop):
    stmt_queue = [(loop,[],[])]
    stmt_info = {}
    while stmt_queue:
        stmt = stmt_queue.pop(0)
        if stmt[0].isentity("loop"):
            info = get_loop_info(stmt[0])
            if not info:
                print("Loop structures are irregular. Cannot deduce dependencies.")
                return None
            order = 0
            for c in stmt[0].body.children:
                if c.isentity("stmt") or c.isentity('expr'):
                    stmt_queue.append((c,stmt[1]+[info],stmt[2]+[order]))
                    order += 1
            continue
        stmt_info[stmt[0].id] = {"loops": stmt[1], "order": stmt[2], "stmt": stmt[0]}
        stmt_info[stmt[0].id]["var_refs"] = stmt[0].query("ref{DeclRefExpr}")
    return stmt_info 

def generate_stmt_instance_identifiers(stmt_info):
    stmt_instance_identifiers = {}
    for stmt_id in stmt_info:
        domain_vars = []
        for i in stmt_info[stmt_id]['loops']:
            domain_vars.append(i['idx'])
        stmt_instance_identifiers[stmt_id] = "S%s[%s]" % (stmt_id,",".join(domain_vars))
    return stmt_instance_identifiers

def check_symbolic_vars(l):
    symbolic_vars = []
    for i in l:
        if not str(i).isnumeric() and not str(i) in symbolic_vars:
            symbolic_vars.append(str(i))
    symbols = ""
    if symbolic_vars:
        symbols = "[%s]->" % (",".join(symbolic_vars))
    return symbols

def eval_const_exprs(expr):
    to_eval = []
    expr_ = expr
    while '(' in expr_:
        st = expr_.index('(')
        e = expr_.index(')')
        to_eval.append(expr_[st:e+1])
        expr_ = expr_[e+1:]
    for e in to_eval:
        try:
            tmp = eval(e)
            expr = expr.replace(e,str(tmp))
        except:
            continue
    return expr

def array_access_relation(stmt_instance_id, ref, stmt_domain, arr_constraint, symbol_map):
    idx = ref.parent.children[1].unparse()
    refs_in_idx = ref.parent.children[1].query('ref{DeclRefExpr}') 
    domain_vars = stmt_instance_id[stmt_instance_id.index('[')+1:stmt_instance_id.index(']')].split(',')
    to_check = []
    for r in refs_in_idx:
        var = r.ref.unparse()
        if var not in domain_vars:
            if var in symbol_map:
                idx = idx.replace(var, symbol_map[var])
            else:
                to_check.append(var)
    symbols = check_symbolic_vars(to_check)
    arr = ref.unparse()
    idx = eval_const_exprs(idx)
    return isl.Map("%s{%s->%s[_i_]: _i_=%s}" % (symbols,stmt_instance_id, arr, idx)).intersect_domain(stmt_domain).intersect_range(arr_constraint)

def scalar_access_relation(stmt_instance_id, var, stmt_domain):
    return isl.Map("{%s->%s[]:}" % (stmt_instance_id, var)).intersect_domain(stmt_domain)

# TODO: currently only support 1D arrays,  
def add_array_constraint(ref, array_constraints):
    arr = ref.unparse()
    if arr in array_constraints:
        return
    if ref.shape.dim:
        (st, end) = (0, ref.shape.dim[0])
    else:
        # TODO: handle unknown size case (symbolic)
        (st, end) = (0, 'N')
    symbols = check_symbolic_vars([st,end])
    array_constraints[arr] = isl.Set("%s{%s[_i_]: %s<=_i_<%s}" % (symbols,arr,st,end))

def determine_access_relations(stmt_info, stmt_instance_ids, stmt_it_domains, local_vars, symbol_map):
    reads = {}
    writes = {}
    array_constraints = {}
    cnt = 0 # counter so multiple accesses to same var in same stmt can be differentiated
    for id in stmt_info:
        accesses = stmt_info[id]['var_refs']
        for row in accesses:
            var = row.ref.unparse()
            if function_type(row.ref.type) or var in local_vars:
                continue
            rw = read_or_write(row.ref)
            if row.ref.parent.isentity('ArraySubscriptExpr') and row.ref == row.ref.parent.children[0]:
                add_array_constraint(row.ref, array_constraints)
                relation = array_access_relation(stmt_instance_ids[id], row.ref, stmt_it_domains[id], array_constraints[row.ref.unparse()], symbol_map) 
            else:
                relation = scalar_access_relation(stmt_instance_ids[id], var, stmt_it_domains[id])
            if 'R' in rw:
                reads[(id, var, cnt)] = relation
            if 'W' in rw:
                writes[(id, var, cnt)] = relation
            cnt += 1
    return reads, writes

def determine_iteration_domains_and_schedule(stmt_info, stmt_instance_ids, symbol_map):
    stmt_iteration_domains = {}
    stmt_schedules = {}
    global_sched_len = 2*max([len(stmt['loops']) for stmt in stmt_info.values()])
    sched_start = "[%s]" % ','.join(['t'+str(i) for i in range(0,global_sched_len)])
    for id in stmt_info:
        loops = stmt_info[id]['loops']
        # iteration domain
        domain_ranges = []
        to_check = []
        for i in loops:
            if i['end'] in symbol_map:
                i['end'] = symbol_map[i['end']]
            if i['start'] in symbol_map:
                i['start'] = symbol_map[i['start']]
            domain_ranges.append(i['start'] + "<=" + i['idx'] + "<" + i['end'])
            to_check.append(i['start'])
            to_check.append(i['end'])
        symbols = check_symbolic_vars(to_check)
        stmt_iteration_domains[id] = isl.Set("%s{%s: %s}" % (symbols, stmt_instance_ids[id]," and ".join(domain_ranges)))
        # schedule
        sched = []
        for i in range(len(loops)):
            sched += [loops[i]['idx']] + [stmt_info[id]['order'][i]]
        sched += [0]*(global_sched_len-len(sched))
        stmt_schedules[id] = isl.Map("{%s->%s: %s}" % (stmt_instance_ids[id], sched_start, ' and '.join(['t'+str(i)+'='+str(sched[i]) for i in range(0,global_sched_len)])))
    return stmt_iteration_domains, stmt_schedules

def generate_isl_unionmap(maps, debug=False):
    if not maps:
        return None
    maps = list(maps.values())
    unionmap = isl.UnionMap(maps[0])
    for m in maps[1:]:
        unionmap = unionmap.union(m)
    return unionmap

def build_fixed_symbol_map(loop):
    results = loop.query('a{ArraySubscriptExpr} => ref{DeclRefExpr}')
    fixed_symbol_map = {}
    done = []
    for row in results:
        if row.ref.name not in done and row.ref.parent != row.a and not (row.ref.decl.parent.isentity('DeclStmt') and row.ref.decl.parent.parent.isentity('ForStmt')):
            refs_in_loop = loop.query('ref{DeclRefExpr}', where=lambda ref: ref.name == row.ref.name)
            if not any([read_or_write(i.ref) != 'R' for i in refs_in_loop]): # read only, not modified in loop
                fixed_symbol_map[row.ref.name] = str(random.randint(1,10))
        done.append(row.ref.name)
    return fixed_symbol_map

def build_polyhedral_model(loop, debug=False):
    local_vars = [v.name for v in get_local_var_list(loop)]
    stmt_info = process_stmts(loop)
    fixed_symbol_map = build_fixed_symbol_map(loop)
    if not stmt_info:
        return None, None, None
    try:
        stmt_instance_ids = generate_stmt_instance_identifiers(stmt_info)
        stmt_iteration_domains, stmt_schedules = determine_iteration_domains_and_schedule(stmt_info, stmt_instance_ids, fixed_symbol_map)
        reads, writes = determine_access_relations(stmt_info, stmt_instance_ids, stmt_iteration_domains, local_vars, fixed_symbol_map)
    except: # unable to build polyhedral model
        return None, None, None
    schedule = generate_isl_unionmap(stmt_schedules)
    reads = generate_isl_unionmap(reads, debug=True)
    writes = generate_isl_unionmap(writes)
    return reads, writes, schedule

def is_parallel(reads, writes, schedule, idx_var, debug=False):
    if not schedule:
        return None
    if not writes or not reads:
        return True
    raw_deps, war_deps, waw_deps = analyse_loop_deps(reads, writes, schedule, debug=debug)
    return is_loop_parallel(raw_deps, war_deps, waw_deps, idx_var)