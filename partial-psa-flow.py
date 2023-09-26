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

def branch_decision(ast, data):
    if 'target' in data:
        return data['target']
    return informed_branch_decision(ast, data)

no_flow = DesignFlow('none')

# construct OpenMP multi-thread CPU design-flow branch
omp_flow = DesignFlow('omp')
omp_flow.add_pattern(multithread_parallel_loops)

# construct oneAPI CPU+FPGA design-flow branch
oneapi_flow = DesignFlow('oneapi')
oneapi_flow.add_pattern(generate_oneapi_design)
oneapi_flow.add_pattern(employ_sp_fp_literals)
oneapi_flow.add_pattern(employ_sp_math_fns)
oneapi_flow.add_pattern(unroll_small_fixed_bound_loops)

# construct HIP CPU+GPU design-flow branch
hip_flow = DesignFlow('hip')
hip_flow.add_pattern(generate_hip_design)
hip_flow.add_pattern(employ_sp_fp_literals)
hip_flow.add_pattern(employ_sp_math_fns)
hip_flow.add_pattern(employ_reciprocal_math_fns)
hip_flow.add_pattern(employ_hip_pinned_memory)
hip_flow.add_pattern(introduce_shared_mem_buffers)

# construct target independent design-flow branch
design_flow = DesignFlow('main')
design_flow.add_pattern(extract_hotspot, {'filter_fn': parallel_filter,'fn_name': 'kernel___','threshold': 0.4})
design_flow.add_pattern(remove_compound_assignment_deps)
design_flow.add_pattern(data_inout_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(loop_tripcount_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(arithmetic_intensity_analysis,{'exec_rule':'orig'})
design_flow.add_pattern(pointer_analysis)
design_flow.add_pattern(loop_dependence_analysis)
design_flow.add_branchpoint(branch_decision, [no_flow, omp_flow, oneapi_flow, hip_flow])

## run the PSA-flow 
usage = ("Usage:\n  artisan partial-psa-flow.py app_name <target>\n"
         "app_name = adpredictor | nbody-sim | bezier-surface | rush-larsen | kmeans\n"
         "target = all | cpu | gpu | fpga ")    

if len(sys.argv) < 2:
    print(usage)
    exit()

app = sys.argv[1]
if app not in ['adpredictor', 'nbody-sim', 'bezier-surface', 'rush-larsen', 'kmeans']:
    print(usage)
    exit()

args = {}
target = "Auto-Selected Target"
if len(sys.argv) > 2:
    if sys.argv[2] == 'all':
        args['target'] = [1,2,3]
        target = "CPU, FPGA, and GPU"
    if sys.argv[2] == 'cpu':
        args['target'] = [1]
        target = "CPU"
    if sys.argv[2] == 'fpga':
        args['target'] = [2]
        target = "FPGA"
    if sys.argv[2] == 'gpu':
        args['target'] = [3]
        target = "GPU"

src = f'cpp_apps/{app}/main.cpp'
dest = f'gen/{app}'
print(f"Running the partial PSA-flow on {app} for {target}...")   
final_ast = design_flow.run(src, dest, args=args)