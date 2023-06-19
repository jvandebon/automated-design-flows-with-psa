#!/usr/bin/env artisan

from meta_cl import *
from profilers import *
from util import *
import numpy as np

def identify_hotspot_loops(ast, threshold, filter_fn=outermost_filter, debug=False):
    profiler = LoopTimeProfiler(ast)
    profiler.run(debug=debug,filter_fn=filter_fn)
    profile = profiler.data
    app_time = profile['main_fn']
    profile.pop('main_fn')
    min_hotspot_time = app_time*threshold
    hotspot_candidates = []
    for loop in profile:
        if profile[loop] > min_hotspot_time:
            hotspot_candidates.append((loop, profile[loop]))
    hotspot_candidates.sort(key=lambda l: l[1], reverse=True)
    return hotspot_candidates

def inline_functions_with_pointer_args(ast, fn):
    to_inline = []
    calls = fn.query('c{CallExpr}')
    for row in calls:
        # TODO: operator calls are caught here - handle 
        if "operator=" in row.c.name:
            continue
        for arg in row.c.args:
            pointer_args = arg.query('r{DeclRefExpr}', where=lambda r: pointer_type(r.type))
            if pointer_args:
                to_inline.append(row.c)
                break
    for call in to_inline:
        inline_fn(ast, fn, call)

def analyse_tripcounts(ast, fn_name, debug=False, exec_rule=''):
    # statically check if any tripcounts are fixed
    # TODO: currently assumes simple loop incremement condition (idx++/idx--)
    fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)[0].fn
    loops = fn.query(select="(loop{ForStmt})")
    fixed_bound_loops = []
    for row in loops:
        info = get_loop_info(row.loop)
        if info['start'].isdigit() and info['end'].isdigit():
            fixed_bound_loops.append(row.loop.tag)
    # dynamically analyse tripcounts 
    profiler = LoopTripCountProfiler(ast, fn_name)
    data = profiler.run(debug=debug, exec_rule=exec_rule)
    # parse profiler results
    tripcounts = {}
    for tag in data:
        tripcounts[tag] = {}
        tripcounts[tag]['total'] = data[tag]['total']
        tripcounts[tag]['instances'] = data[tag]['instances']
        tripcounts[tag]['average'] = int(data[tag]['total']/data[tag]['instances'])
        if tag in fixed_bound_loops:
            tripcounts[tag]['fixed'] = True
        else:
            tripcounts[tag]['fixed'] = False
    return tripcounts

def run_data_inout_analysis(ast, fn_name, debug=False, exec_rule=''):
    profiler = DataInOutProfiler(ast)
    profiler.run(fn_name,debug=debug, exec_rule=exec_rule)
    return profiler.data

def memory_footprint_analysis(ast, fn_name, debug=False, exec_rule=''):
    profiler = MemoryFootprintProfiler(ast)
    profiler.run(fn_name, debug=debug, exec_rule=exec_rule)
    return profiler.data

math_fn_to_flops = {'sin': 2, 'cos': 2, 'tan': 2, 'sincos': 2, 'log': 2, 'log2': 2, 'log10': 2, 'exp': 2, 'exp2': 2, 'exp10': 2, 'pow': 2, 'sqrt': 2, 'rsqrt' : 2, 'fabs': 2, 'sinf': 2, 'cosf': 2, 'tanf': 2, 'sincosf': 2, 'logf': 2, 'log2f': 2, 'log10f': 2, 'expf': 2, 'exp2f': 2, 'exp10f': 2, 'powf': 2, 'sqrtf': 2, 'rsqrtf' : 2, 'fabsf': 2, 'floor': 2}
def count_flops_basecase(scope, tripcounts, fn_flop_map):
    # count basic fp ops (+, -, /, *, ...)
    ops = scope.query("(uop{UnaryOperator}|bop{BinaryOperator}|cop{CompoundAssignmentOperator})")
    flop_count = 0
    for row in ops:
        op = row.uop if row.uop else row.bop if row.bop else row.cop
        if len(op.children) < 2 or op.symbol == '=' or op.type.spelling == 'bool':
            continue
        if op.type.spelling == 'float' or op.type.spelling == 'double':
            flop_count += 1
    # count ops based on calls to math fns 
    math_fn_calls = scope.query('c{CallExpr}', where=lambda c: c.name in math_fn_to_flops)
    for row in math_fn_calls:
        flop_count += math_fn_to_flops[row.c.name]
    # count ops in any called fns (recursively check called fns, build map for quick lookup)
    other_fn_calls = scope.query('c{CallExpr}', where=lambda c: c.name  not in math_fn_to_flops and c.name != 'operator=')
    for row in other_fn_calls:
        fn_name = row.c.name
        if not fn_name:
            continue
        if fn_name not in fn_flop_map:
            fn = scope.ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)
            if fn:
                fn_count = count_flops(fn[0].fn.body, 0, tripcounts, fn_flop_map)
                fn_flop_map[fn_name] = fn_count
            else:
                print("Can't count flops for %s, can't find function decl." % fn_name)
                continue
        flop_count += fn_flop_map[fn_name]   
    return flop_count 

def count_flops(scope, count, tripcounts, fn_flop_map):
    for child in scope.children:
        if child.isentity('ForStmt') or child.isentity('WhileStmt'):
            if child.tag in tripcounts:
                trip_count = tripcounts[child.tag]['average']
                count += trip_count * count_flops(child.body, 0, tripcounts, fn_flop_map)
            else:
                print("Can't determine trip count of loop %s, skipping." % child.location)
        else:
            count += count_flops_basecase(child, tripcounts, fn_flop_map)
    return count

def calculate_arithmetic_intensity(ast, fn_name, tripcounts=None, memory_footprint=None, exec_rule=''):
    if not tripcounts:
        tripcounts = loop_tripcount_analysis(ast, fn_name, exec_rule=exec_rule)
    # recursively count flops
    results = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)
    if not results:
        return 
    flops = count_flops(results[0].fn.body, 0, tripcounts, {})
    # get memory footprint
    if not memory_footprint:
        memory_footprint = memory_footprint_analysis(ast, fn_name, exec_rule=exec_rule)
    bytes_accessed = memory_footprint['__TOTAL__']['bytes_R'] + memory_footprint['__TOTAL__']['bytes_W']
    # calculate and return flops/B
    return {'flops': flops, 'bytes': bytes_accessed, 'intensity': float(flops/bytes_accessed)}

def analyse_loop_dependencies(ast, fn_name):
    loops = ast.query(select="fn{FunctionDecl}=>l{ForStmt}", where=lambda fn: fn.name == fn_name)
    loop_deps = {}
    for row in loops:
        reads, writes, schedule = build_polyhedral_model(row.l)
        if not schedule:
            return
        loop_deps[row.l.tag] = {}
        raw_deps, war_deps, waw_deps = analyse_loop_deps(reads, writes, schedule)
        loop_deps[row.l.tag]['raw'] = raw_deps
        loop_deps[row.l.tag]['war'] = war_deps
        loop_deps[row.l.tag]['waw'] = waw_deps
        loop_deps[row.l.tag]['parallel'] = is_loop_parallel(raw_deps, war_deps, waw_deps,get_loop_info(row.l)['idx'])
        if not writes or not reads:
            loop_deps[row.l.tag]['parallel'] = True    
    return loop_deps

def pointer_alias_analysis(ast, fn_name, debug=False):
    # TODO: handle memcpy accesses
    # find the function and associated pointer parameters
    fn = ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == fn_name)
    if not fn or len(fn) > 1:
        exit(1)
    fn = fn[0].fn
    pointer_params = [p[1] for p in fn.signature.params if pointer_type(p[0])]
    # instrument and execute code to trace all pointers in specified function
    profiler = PointerRangeProfiler(ast, fn.name)
    profiler.run(debug=debug)
    points_to_map = build_points_to_map(profiler.data)
    # check if any pointer parameters may alias according to the points-to map 
    pairs = [(p1,p2) for p1 in pointer_params for p2 in pointer_params if p1 != p2]
    aliasing_pairs = [is_reachable(p[0],p[1],points_to_map) for p in pairs]
    alias_pairs = [pairs[idx] for idx in [i for i, x in enumerate(aliasing_pairs) if x == True]]

    # return pairs of parameters that may alias
    return alias_pairs

def remove_loop_arr_deps(ast, kernel_fn, arr_var):
    loops = [row.loop for row in ast.query("f{FunctionDecl} => loop{ForStmt}", where=lambda f: f.name == kernel_fn)]
    loops.sort(key=lambda l: l.depth, reverse=True)
    depth = loops[0].depth
    refs = ast.query("f{FunctionDecl} => loop{ForStmt} => arr{ArraySubscriptExpr}", where=lambda f, arr, loop: f.name == kernel_fn and arr.children[0].name == arr_var and loop.depth == depth)
    while refs:
        remove_loop_arr_dep_with_scalar(refs[0].loop, refs[0].arr)
        ast.commit()
        refs = ast.query("f{FunctionDecl} => loop{ForStmt} => arr{ArraySubscriptExpr}", where=lambda f, arr, loop: f.name == kernel_fn and arr.children[0].name == arr_var and loop.depth == depth)

#TODO: currently assumes only one reference in loop (check if var decl already exists)
def remove_loop_arr_dep_with_scalar(loop, arr_ref):
    if arr_ref.parent.isentity('MemberRefExpr'):
        arr_ref = arr_ref.parent
    new_var = "_"+arr_ref.unparse().replace("]","").replace("[","").replace(".","")+"_"
    # initialise new scalar variable
    initialise_var = f"{arr_ref.type.spelling} {new_var} = {arr_ref.unparse()};\n"
    loop.instrument(Action.before, code=initialise_var)
    # replace references to var in loop
    arr_ref.instrument(Action.replace, code=new_var)
    # assign value back to array variable
    assign_value = f"\n{arr_ref.unparse()} = {new_var};"
    loop.instrument(Action.after, code=assign_value)

def use_sp_math_functions(ast, fns):
    math_funcs = ['sin', 'cos', 'tan', 'sincos', 'log', 'log2', 'log10', 'exp', 'exp2', 'exp10', 'pow', 'sqrt', 'rsqrt', 'fabs']
    calls = ast.query('fn{FunctionDecl} => c{CallExpr}', where=lambda c,fn: c.name in math_funcs and fn.name in fns)
    while calls:
        call_descendents = [[des.id for des in row.c.descendants] for row in calls]  
        call_descendents = [item for sublist in call_descendents for item in sublist]                   
        for call in calls:  
            if call.c.id in call_descendents:
                continue
            arg_string = ",".join([arg.unparse() for arg in call.c.args])
            call.c.instrument(Action.replace, code=f"{call.c.name}f({arg_string})")
        ast.commit()
        calls = ast.query('fn{FunctionDecl} => c{CallExpr}', where=lambda c,fn: c.name in math_funcs and fn.name in fns)


def use_sp_fp_literals(ast, fns):
    funcs = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name in fns)
    for row in funcs:
        func = row.fn
        fp_literals = func.body.query('l{FloatingLiteral}', where=lambda l: l.unparse().strip()[-1] != 'f')
        for row in fp_literals:
            row.l.instrument(Action.replace, code=row.l.unparse().strip()+'f')
        int_literals = func.body.query("il{IntegerLiteral}", where=lambda il: il.parent.type.spelling == 'float' or il.parent.type.spelling == 'double')
        for row in int_literals:
            row.il.instrument(Action.replace, code=f"{row.il.unparse()}.0f")

def add_parameter(params, ref):
    params.append({'name': ref.name, 'type': ref.type.spelling})

def derive_parameter_list(loop):
    var_refs = get_all_referenced_vars(loop)
    loop_local_vars = [v.name for v in get_local_var_list(loop)]
    global_vars = [v.name for v in get_global_var_refs([var for var in var_refs if var.name not in loop_local_vars])]
    params = []
    for ref in var_refs:
        if ref.name not in loop_local_vars and ref.name not in global_vars and ref.name not in [p['name'] for p in params]:
            add_parameter(params, ref)
    return params

def refine_parameter_list(loop, fn, params):
    var_decls_tocopy = ""
    for param in params:
        # find references to the parameter variable within the current function
        decl_refs = fn.query("ref{DeclRefExpr}", where=lambda ref: ref.name == param['name'] and fn.body.encloses(ref))
        decl = decl_refs[0].ref.decl
        # check if parameter variable is declared inside the function and not used outside the loop
        declared_in_fn = fn.body.encloses(decl)
        init = None
        if declared_in_fn:
            # find init
            init = [row.ref.parent for row in decl_refs if (not loop.encloses(row.ref) and row.ref.parent.isentity('BinaryOperator') and row.ref.parent.symbol == '=')]
            print([r.unparse() for r in init])
        if init:
            init = init[0]
        used_outside_loop = any([not row.ref.parent == init and not loop.encloses(row.ref) for row in decl_refs])
        if declared_in_fn and not used_outside_loop:
            # copy variable declaration to new function
            var_decls_tocopy += decl.unparse() + ";\n" + init.unparse() + ";\n"
            # check if any variables referenced in declaration need to be added as parameters
            refs_in_decl = decl.query("ref{DeclRefExpr}", where=lambda ref: ref.name not in [p['name'] for p in params] and ref.decl.is_local)
            refs_in_init = init.query("ref{DeclRefExpr}", where=lambda ref: ref.name not in [p['name'] for p in params] and ref.decl.is_local)
            for row in refs_in_decl:
                add_parameter(params, row.ref)
            for row in refs_in_init:
                add_parameter(params, row.ref)
            # remove existing decl
            decl.instrument(Action.replace, code="")
            decl.instrument(Action.remove_semicolon)
            init.instrument(Action.replace, code="")
            init.instrument(Action.remove_semicolon)
            # remove from param list
            params.remove(param)
    return var_decls_tocopy, params

def extract_loop_to_function(ast, ltag, new_fn_name='__loop_fn__'):
    # query the ast for the relevant loop and containing function
    results = ast.query('fn{FunctionDecl} => l{ForStmt}', where=lambda l: l.tag == ltag)
    if not results or len(results) > 1:
        return 
    loop = results[0].l 
    fn = results[0].fn
    # instrument app to trace memory allocations so we can pass pointer sizes to new function
    trace_memory(ast, scopes_to_ignore=[loop.body])
    # determine parameters for the new function: any referenced variables 
    # that are (1) defined outside of the loop and (2) not global
    params = derive_parameter_list(loop)
    # refine parameters: identify any variables defined locally within the 
    # function scope and only used within the loop, move these decls into
    # the new function and remove them from the initial parameter list
    var_decls_tocopy, params = refine_parameter_list(loop, fn, params)
    # add a parameter p_size for every pointer parameter p
    pointer_size_params = []
    pointer_size_args = []
    for p in [i for i in params if '*' in i['type']]:
        pointer_size_params.append("int %s_size_" % p['name'])
        pointer_size_args.append("Mem::size(%s)" % p['name'])
    # instrument to insert new function 
    param_string =  ", ".join([p['type'] + " " + p['name'] for p in params] + pointer_size_params)
    new_func_def = "void %s(%s){\n%s%s\n}" % (new_fn_name,param_string,var_decls_tocopy,loop.unparse())
    fn.instrument(Action.before, code=new_func_def)
    # instrument to replace original loop with call to new function
    call_args_string = ", ".join([p['name'] for p in params] + pointer_size_args)
    new_func_call = "%s(%s);" % (new_fn_name,call_args_string)
    loop.instrument(Action.replace, code=new_func_call)

def openmp_multithread_loops(ast, scope):
    parallel_loops = scope.query('loop{ForStmt}', where=lambda loop: loop.is_outermost() and parallel_filter(loop))
    srcs = []
    for row in parallel_loops:
        # instrument loop with pragma omp parallel for num_threads(NUM_THREADS)
        row.loop.instrument(Action.before, code="#pragma omp parallel for num_threads(NUM_THREADS)\n")
        srcs.append(row.loop.module)
    # instrument src with #include <omp> and #define NUM_THREADS 8
    srcs = list(set(srcs))
    for src in srcs:
        src.module.instrument(Action.before, code="#include <omp.h>\n#define NUM_THREADS 8\n")
    ast.commit()

def run_openmp_num_threads_DSE(ast, max_threads):
    best = n = 2
    min_time = 100000000000
    while n <= max_threads:
        set_num_threads_omp(ast, n)
        ast.sync(commit=True)
        times = []
        for i in range(0,3): # 3 runs for each n threads
            profiler = LoopTimeProfiler(ast)
            profiler.run(debug=True,filter_fn=omp_parallel_loop_filter, exec_rule='run_omp')
            del profiler.data['main_fn']
            times.append(sum(list(profiler.data.values())))
        if np.mean(times) < min_time:
            min_time = np.mean(times)
            best = n
        n *= 2
    set_num_threads_omp(ast, best)
    ast.sync(commit=True)  

def set_threads_to(pragma):
    if pragma[0:4] == "omp parallel for num_threads":
        return "#pragma omp parallel for num_threads(NUM_THREADS)"
    return True

def set_num_threads_omp(ast, n):
    ast.parse_pragmas()
    def_stmt = ast.query('d{MacroDefinition}', where=lambda d: 'NUM_THREADS' in d.unparse())
    if not def_stmt:
        ast.sources[0].module.instrument(Action.before, code=f"#define NUM_THREADS {n}")
    else:
        def_stmt[0].d.instrument(Action.replace, code=f"NUM_THREADS {n}")
    ast.sources[0].module.instrument(Action.pragmas, fn=set_threads_to)

