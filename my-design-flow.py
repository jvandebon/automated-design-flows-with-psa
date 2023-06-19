from design_flow import *
import pprint
pp = pprint.PrettyPrinter(indent=2)

def branch_decision(ast, data):
    return [2]
    outer_loop = ast.query(select="fn{FunctionDecl}=>l{ForStmt}", where=lambda fn, l: fn.name == data['hotspot_fn_name'] and l.is_outermost())[0].l
    if not data['loop_dep_report'][outer_loop.tag]['parallel']:   # outer loop is not parallel
        if data['arith_intensity_report']['intensity'] > 0.5:
            print("CPU+FPGA")
            return [2]
        else:
            print("SINGLE THREAD CPU")
            return [0]
    else:   # outer loop is parallel
        tripcounts = data['tripcount_report']
        if data['arith_intensity_report']['intensity'] > 0.5:
            if len(tripcounts) > 1:     # if any inner loops
                unrollable_inner_loops = [i for i in tripcounts if i != outer_loop.tag and not tripcounts[i]['fixed']]
                if unrollable_inner_loops:
                    print("CPU+GPU")
                    return [3]
                else:
                    print("CPU+FPGA")
                    return [2]
            else:   # no inner loops 
                print("CPU+FPGA, CPU+GPU")
                return [2,3]
        else:
            print("MULTI-THREAD CPU")
            return [1]

def gpu_decision(ast, data):
    return [0,1]

def fpga_decision(ast, data):
    return [0,1]

no_flow = design_flow('none')

omp_flow = design_flow('omp')
omp_flow.add_pattern(multithread_parallel_loops)
omp_flow.add_pattern(omp_nthreads_dse)

arria10_flow = design_flow('a10')
arria10_flow.add_pattern(unroll_until_fpga_overmap, {'target':'a10'})

stratix10_flow = design_flow ('s10')
stratix10_flow.add_pattern(use_oneapi_zerocopy_memory)
# TODO: unroll until overmap 
# arria10_flow.add_pattern(unroll_until_fpga_overmap,target='s10')

oneapi_flow = design_flow('oneapi')
oneapi_flow.add_pattern(generate_oneapi_design)
oneapi_flow.add_pattern(employ_sp_fp_literals)
oneapi_flow.add_pattern(employ_sp_math_fns)
oneapi_flow.add_pattern(unroll_small_fixed_bound_loops)
oneapi_flow.add_branchpoint(fpga_decision, [arria10_flow, stratix10_flow])

ti2080_flow = design_flow('ti2080')
ti2080_flow.add_pattern(hip_blocksize_timing_DSE, {'device':'0'})

ti1080_flow = design_flow('ti1080')
ti1080_flow.add_pattern(hip_blocksize_timing_DSE, {'device':'2'})

hip_flow = design_flow('hip')
hip_flow.add_pattern(generate_hip_design)
hip_flow.add_pattern(employ_sp_fp_literals)
hip_flow.add_pattern(employ_sp_math_fns)
hip_flow.add_pattern(employ_reciprocal_math_fns)
hip_flow.add_pattern(employ_hip_pinned_memory)
hip_flow.add_pattern(introduce_shared_mem_buffers)
hip_flow.add_branchpoint(gpu_decision, [ti2080_flow, ti1080_flow])

my_design_flow = design_flow('main')
my_design_flow.add_pattern(extract_hotspot, {'filter_fn': parallel_filter,'fn_name': 'kernel___','threshold': 0.4})
my_design_flow.add_pattern(remove_compound_assignment_deps)
my_design_flow.add_pattern(data_inout_analysis,{'exec_rule':'orig'})
my_design_flow.add_pattern(loop_tripcount_analysis,{'exec_rule':'orig'})
my_design_flow.add_pattern(arithmetic_intensity_analysis,{'exec_rule':'orig'})
my_design_flow.add_pattern(pointer_analysis)
my_design_flow.add_pattern(loop_dependence_analysis)
my_design_flow.add_branchpoint(branch_decision, [no_flow, omp_flow, oneapi_flow, hip_flow])

# app = 'adpredictor'
app = 'nbody-sim'
# app = 'bezier-surface' 
# app = 'rush-larsen' 
# app = 'kmeans'  

src = f'cpp_apps/{app}/main.cpp'
dest = f'gen/{app}'

final_ast = my_design_flow.run(src, dest)
pp.pprint(my_design_flow.data)

