#!/usr/bin/env artisan

from util import *
from oneapi import *
from hip import *
from meta_cl import *
from metaprograms import *


def extract_hotspot(ast, data, filter_fn=outermost_filter, threshold=0.5, fn_name='__kernel__'):
    candidate_loops = identify_hotspot_loops(ast, threshold, filter_fn=filter_fn)
    data['hotspot_fn_name'] = fn_name
    extract_loop_to_function(ast, candidate_loops[0][0], new_fn_name=fn_name)
    ast.sync(commit=True)
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == fn_name)[0].fn
    inline_functions_with_pointer_args(ast, kernel_fn)
    kernel_fn_new = ast.commit(track=[kernel_fn])[0]
    data['device_fns'] = [fn_name] + [fn[0] for fn in get_called_fns(ast, kernel_fn_new)]

def loop_tripcount_analysis(ast, data, debug=False, exec_rule=''):
    tripcounts = analyse_tripcounts(ast, data['hotspot_fn_name'], debug=debug, exec_rule=exec_rule)
    data['tripcount_report'] = tripcounts

def arithmetic_intensity_analysis(ast, data, exec_rule=''):
    tripcounts = None
    if 'tripcount_report' in data:
        tripcounts = data['tripcount_report']
    ai = calculate_arithmetic_intensity(ast, data['hotspot_fn_name'], tripcounts, exec_rule=exec_rule)
    data['arith_intensity_report'] = ai

def pointer_analysis(ast, data, *args):
    alias_pairs = pointer_alias_analysis(ast, data['hotspot_fn_name'])
    data['pointer_alias_report'] = {'alias_pairs': alias_pairs, 'restrict': not len(alias_pairs)}

def data_inout_analysis(ast, data, debug=False, exec_rule=''):
    data_inout = run_data_inout_analysis(ast, data['hotspot_fn_name'], debug=debug, exec_rule=exec_rule)
    data['data_inout_report'] = data_inout

def loop_dependence_analysis(ast, data, *args):
    deps = analyse_loop_dependencies(ast, data['hotspot_fn_name'])
    data['loop_dep_report'] = deps

def generate_hip_design(ast, data):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    map_to_hip_gpu(ast, kernel_fn, restrict=data['pointer_alias_report']['restrict'])
    ast.sync(commit=True)

def employ_sp_fp_literals(ast, data):
    use_sp_fp_literals(ast, data['device_fns'])
    ast.sync(commit=True)

def employ_sp_math_fns(ast, data):
    use_sp_math_functions(ast, data['device_fns'])
    ast.sync(commit=True)

def employ_reciprocal_math_fns(ast, data):
    use_reciprocal_math_functions(ast, data['device_fns'])
    ast.sync(commit=True)

def employ_hip_pinned_memory(ast, data):
    use_pinned_memory(ast)
    ast.sync(commit=True)

def hip_blocksize_timing_DSE(ast, data, device=None):
    time_kernel_bs_DSE(ast, data['hotspot_fn_name'], device=device)

def multithread_parallel_loops(ast, data):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    openmp_multithread_loops(ast, kernel_fn)
    ast.sync(commit=True)

def remove_compound_assignment_deps(ast, data, *args):
    dep_refs = ast.query('fn{FunctionDecl} => l{ForStmt} => pe{CompoundAssignmentOperator} => arr{ArraySubscriptExpr}', where=lambda fn, pe, arr: fn.name == 'kernel___' and pe.children[0].encloses(arr))
    dep_vars = []
    for row in dep_refs:
        dep_vars.append(row.arr.children[0].name)
    dep_vars = list(set(dep_vars))
    for var in dep_vars:
        remove_loop_arr_deps(ast, data['hotspot_fn_name'], var)
        ast.sync(commit=True)

def omp_nthreads_dse(ast, data, max_threads=32):
    run_openmp_num_threads_DSE(ast, max_threads)

def generate_oneapi_design(ast, data, zerocopydata=False):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    if zerocopydata:
        map_to_oneapi_zerocopydatatransfer(ast, kernel_fn, data['pointer_alias_report']['restrict'], kernel_fn.name, device_ptrs=[])
    else:
        map_to_oneapi_basic(ast, kernel_fn, data['pointer_alias_report']['restrict'], kernel_fn.name)
    ast.sync(commit=True)

def use_oneapi_zerocopy_memory(ast,data):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    basic_kernel_to_zerocopy(ast,kernel_fn)
    ast.sync(commit=True)

def unroll_small_fixed_bound_loops(ast, data, max_iters=20):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    unroll_fixed_oneapi_loops(ast, kernel_fn, max_iters=max_iters)
    ast.sync(commit=True)


def introduce_shared_mem_buffers(ast, data, param=None, max_size=10000):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    wrapper_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == f"{kernel_fn.name}_wrapper_")[0].fn
    introduce_shared_mem(ast, kernel_fn, wrapper_fn, data['data_inout_report'], data['struct_map'], max_size=max_size)


# IN PROGRESS BELOW:

def unroll_until_fpga_overmap(ast,data,target='a10'):
    kernel_fn = ast.query('fn{FunctionDecl}', where=lambda fn: fn.name == data['hotspot_fn_name'])[0].fn
    unroll_until_overmap_dse(ast, kernel_fn, target=target)     