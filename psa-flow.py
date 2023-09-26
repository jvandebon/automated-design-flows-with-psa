from design_flow import *
import sys
import pprint
import copy
pp = pprint.PrettyPrinter(indent=2)

def informed_branch_decision(ast, data):
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

def uninformed_branch_decision(ast, data):
    return [0,1,2]

def gpu_decision(ast, data):
    return [0,1]

def fpga_decision(ast, data):
    return [0,1]

no_flow = DesignFlow('none')

# construct OpenMP multi-thread CPU design-flow branch
omp_flow = DesignFlow('omp')
omp_flow.add_pattern(multithread_parallel_loops)
omp_flow.add_pattern(omp_nthreads_DSE)

# construct Arria10 CPU+FPGA design-flow branch
arria10_flow = DesignFlow('a10')
arria10_flow.add_pattern(unroll_until_fpga_overmap_DSE, {'target':'a10'})

# construct Stratix10 CPU+FPGA design-flow branch
stratix10_flow = DesignFlow('s10')
stratix10_flow.add_pattern(use_oneapi_zerocopy_memory)
stratix10_flow.add_pattern(unroll_until_fpga_overmap_DSE, {'target':'s10'})

# construct oneAPI CPU+FPGA design-flow branch
oneapi_flow = DesignFlow('oneapi', 'oneapi_fpga')
oneapi_flow.add_pattern(generate_oneapi_design)
oneapi_flow.add_pattern(employ_sp_fp_literals)
oneapi_flow.add_pattern(employ_sp_math_fns)
oneapi_flow.add_pattern(unroll_small_fixed_bound_loops)
oneapi_flow.add_branchpoint(fpga_decision, [arria10_flow, stratix10_flow])

# construct RTX 2080 Ti CPU+GPU design-flow branch
ti2080_flow = DesignFlow('ti2080')
# 'device' parameter indicates CUDA_VISIBLE_DEVICE required for Ti 2080
ti2080_flow.add_pattern(hip_blocksize_timing_DSE, {'device':'0'})

# construct GTX 1080 Ti CPU+GPU design-flow branch
ti1080_flow = DesignFlow('ti1080')
# 'device' parameter indicates CUDA_VISIBLE_DEVICE required for Ti 1080
ti1080_flow.add_pattern(hip_blocksize_timing_DSE, {'device':'2'})

# construct HIP CPU+GPU design-flow branch
hip_flow = DesignFlow('hip', 'hip_gpu')
hip_flow.add_pattern(generate_hip_design)
hip_flow.add_pattern(employ_sp_fp_literals)
hip_flow.add_pattern(employ_sp_math_fns)
hip_flow.add_pattern(employ_reciprocal_math_fns)
hip_flow.add_pattern(employ_hip_pinned_memory)
hip_flow.add_pattern(introduce_shared_mem_buffers)
hip_flow.add_branchpoint(gpu_decision, [ti2080_flow, ti1080_flow])

# construct target independent design-flow branch
design_flow = DesignFlow('main')
design_flow.add_pattern(extract_hotspot, {'filter_fn': parallel_filter,'fn_name': 'kernel___','threshold': 0.4})
design_flow.add_pattern(remove_compound_assignment_deps)
design_flow.add_pattern(data_inout_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(loop_tripcount_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(arithmetic_intensity_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(pointer_analysis)
design_flow.add_pattern(loop_dependence_analysis)

# two design-flow versions: informed, uninformed
informed_design_flow = copy.deepcopy(design_flow)
informed_design_flow.add_branchpoint(informed_branch_decision, [no_flow, omp_flow, oneapi_flow, hip_flow])
uninformed_design_flow = design_flow
uninformed_design_flow.add_branchpoint(uninformed_branch_decision, [omp_flow, oneapi_flow, hip_flow])

## run the PSA-flows
usage = ("Usage:\n  artisan psa-flow.py app_name <uninformed(optional)>\n"
         "app_name = adpredictor | nbody-sim | bezier-surface | rush-larsen | kmeans")

if len(sys.argv) < 2:
    print(usage)
    exit()

app = sys.argv[1]
if app not in ['adpredictor', 'nbody-sim', 'bezier-surface', 'rush-larsen', 'kmeans']:
    print(usage)
    exit()

informed = True
if len(sys.argv) > 2 and sys.argv[2] != "uninformed":
    print(usage)
    exit()
elif len(sys.argv) > 2:
    informed = False

src = f'cpp_apps/{app}/main.cpp'
dest = f'gen/{app}'

if informed:
    print(f"Running the informed PSA-flow on {app}...")
    final_ast = informed_design_flow.run(src, dest)
else:
    print(f"Running the uninformed PSA-flow on {app}...")
    final_ast = uninformed_design_flow.run(src, dest)

# pp.pprint(informed_design_flow.data)