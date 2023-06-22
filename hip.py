from meta_cl import *
from util import *
from metaprograms import extract_loop_to_function
import numpy as np
import os 

def get_kernel_signature(fn, restrict):
     # currently checks that all pointers are alias free,
     # could check individually for aliasing
    params = fn.signature.params
    new_params = []
    if restrict:
        for p in params:
            p_type = p[0]
            p_name = p[1]
            if pointer_type(p_type):
                new_params.append("%s __restrict__ %s" % (p_type.spelling, p_name))
            else:
                new_params.append("%s %s" % (p_type.spelling, p_name))
    else:
        new_params = ["%s %s" % (p[0].spelling, p[1]) for p in params]
    return "__global__ %s %s(%s)" % (fn.signature.rettype.spelling, fn.name, ','.join(new_params))

def next_multiple(a, b):
    return (a + b - 1) // b * b

# only supports 1D for now, gid = threadIdx.x
def generate_hip_kernel(ast, fn, restrict):
    # __global__ kernel signature with restricted pointers if applicable
    kernel_signature = get_kernel_signature(fn,restrict)
    # check helper functions -- add __device__ __host__
    called_fns = get_called_fns(ast, fn)
    for f in called_fns:
        name = f[0]
        decl = f[1]
        decl.instrument(Action.before, code="__device__ __host__ ")
    loop = fn.query('l{loop}', where=lambda l: l.is_outermost())[0].l
    problem_size = get_loop_info(loop)['end']
    # remove outer loop, set loop idx var to global id (threadIdx.x)
    gid_varinit = "unsigned %s = blockDim.x*blockIdx.x+threadIdx.x;;" % get_loop_info(loop)['idx']   
    # check that global id is within problem size bounds 
    bounds_check = "if (%s >= %s) return;" % (get_loop_info(loop)['idx'] , problem_size)
    # generate kernel function string
    kernel_body = "%s\n%s\n%s" % (gid_varinit, bounds_check, loop.body.unparse()[1:-1])
    kernel_fn  = "%s {\n%s\n}" % (kernel_signature, kernel_body)
    return problem_size, kernel_fn

def map_to_hip_gpu(ast, fn, restrict=False):
    # add relevant includes
    fn.module.instrument(Action.before, code="#include \"hip/hip_runtime.h\"\n")
    # create and hipmalloc gpu pointers, hipmemcpy read only vars from host to device,
    # hipmemcpy write only vars from device to host, hipFree new pointers
    pointers = [] 
    params = fn.query('p{ParmDecl}')
    param_names = [row.p.name for row in params]
    wrapper_fn_pointer_decls = ""
    wrapper_fn_hipmalloc = ""
    wrapper_fn_hipfree = ""
    wrapper_fn_memcpy_todevice = ""
    wrapper_fn_memcpy_tohost = ""
    hipmalloc_template = "hipMalloc((void**)&%s, %s);\n"
    memcpy_template = "hipMemcpy(%s, %s, %s, %s);\n"
    args = []
    for row in params:
        param = row.p
        if pointer_type(param.type):
            size = "%s_size_" % param.name
            if not size in param_names:
                print("Can't determine size of pointer: %s, exiting..." % param)
                exit(1)
            rw = read_or_write_param(param, fn)
            wrapper_fn_pointer_decls += param.unparse() + "_gpu;\n"
            wrapper_fn_pointer_decls = wrapper_fn_pointer_decls.replace("const", "")
            wrapper_fn_hipmalloc += hipmalloc_template  % (param.name+"_gpu", size)
            wrapper_fn_hipfree += "hipFree(%s_gpu);\n" % param.name
            
            if 'R' in rw:
                wrapper_fn_memcpy_todevice += memcpy_template % (param.name+"_gpu", param.name, size, "hipMemcpyHostToDevice")
            if 'W' in rw:
                wrapper_fn_memcpy_tohost += memcpy_template % (param.name, param.name+"_gpu", size, "hipMemcpyDeviceToHost")
            args.append(param.name + "_gpu")
        else:
            args.append(param.name)
    # transform function to __global__ HIP kernel
    problem_size, kernel_fn = generate_hip_kernel(ast, fn, restrict)
    # insert code to print device information
    device_check = "hipDeviceProp_t prop;\nhipGetDeviceProperties(&prop, 0);\nprintf(\"\\nDevice: %s\\n\", prop.name);\n"
    # launch kernel (need problem size, block size, kernel args)
    launch_template = "size_t dyn_shared = 0;\nhipLaunchKernelGGL(%s, dim3(%s/_blocksize_), dim3(_blocksize_), dyn_shared, 0, %s);\n"
    launch_string = launch_template % (fn.name, "next_multiple(%s, _blocksize_)" % problem_size, ', '.join(args))
    blocksize = "const int _blocksize_ = 256;\n"
    next_multiple_string = "unsigned long int next_multiple(int a, int b) { return (unsigned long)((a + b - 1) / b * b); }\n"
    # if kernel call is inside a loop -- extract outer loop to wrapper function 
    called_in_loop = ast.query('fn{FunctionDecl} => loop{ForStmt} => call{CallExpr}', where=lambda call: call.name == fn.name)
    if len(called_in_loop) > 0:
        loop = called_in_loop[0].loop
        extract_loop_to_function(ast, loop.tag, new_fn_name=f"{fn.name}_wrapper_")
        ast.commit()
        # instrument new wrapper function with all hipmalloc / hipmemcpy / hipfree code 
        wrapper_fn = ast.query('w_fn{FunctionDecl}', where=lambda w_fn: w_fn.name == f"{fn.name}_wrapper_")[0].w_fn         
        wrapper_fn.instrument(Action.before, code=f"{next_multiple_string}\n") #{blocksize}\n")
        wrapper_fn.instrument(Action.begin, code=f"{device_check}\n{blocksize}\n{wrapper_fn_pointer_decls}\n{wrapper_fn_hipmalloc}\n{wrapper_fn_memcpy_todevice}\n")
        wrapper_fn.instrument(Action.end, code=f"{wrapper_fn_memcpy_tohost}\n{wrapper_fn_hipfree}\n")
        ast.commit()
        # replace existing kernel function with global version
        kernel = ast.query('kfn{FunctionDecl}', where=lambda kfn: kfn.name == fn.name)[0].kfn 
        kernel.instrument(Action.replace, code=f"{kernel_fn}\n")
        # replace call to kernel with launch string 
        kernel_call = ast.query('w_fn{FunctionDecl} => call{CallExpr}', where=lambda w_fn, call: w_fn.name == f"{fn.name}_wrapper_" and call.name == fn.name)[0].call
        kernel_call.instrument(Action.replace, code=launch_string)
        ast.commit()
    else:
        # create new wrapper function
        loc = str(fn.signature).index("(")
        wrapper_fn_sig = "%s %s_wrapper_%s" % (str(fn.signature)[:loc], fn.name,str(fn.signature)[loc:])
        wrapper_fn_string = f"{next_multiple_string}{wrapper_fn_sig}{{\n{device_check}\n{blocksize}\n{wrapper_fn_pointer_decls}\n{wrapper_fn_hipmalloc}\n{wrapper_fn_memcpy_todevice}\n{launch_string}\n{wrapper_fn_memcpy_tohost}\n{wrapper_fn_hipfree}\n}}\n"
        # insert wrapper function and new kernel, replacing original function
        fn.instrument(Action.replace, code="%s\n%s" % (kernel_fn, wrapper_fn_string))
        # replace calls to original function with call to wrapper
        calls = ast.query('c{CallExpr}', where=lambda c: c.name == fn.name)
        for row in calls:
            row.c.instrument(Action.replace, code=row.c.unparse().replace(fn.name, fn.name+"_wrapper_"))
    ast.commit()
    return

def use_reciprocal_math_functions(ast, device_functions):
    funcs_with_reciprocals = ['sqrt','cbrt','sqrtf','cbrtf']
    calls = ast.query('fn{FunctionDecl} => c{CallExpr}', where=lambda c,fn: c.name in funcs_with_reciprocals and fn.name in device_functions and c.parent.isentity("BinaryOperator") and c.parent.symbol == "/" and c.id == c.parent.children[1].id)
    for call in calls:
        new_code = call.c.parent.children[0].unparse() + " * " + "r"+call.c.unparse()
        call.c.parent.instrument(Action.replace, code=new_code)

def use_pinned_memory(ast):
    mallocs = ast.query('c{CallExpr}', where=lambda c: c.name == 'malloc')
    pinned_pointers = []
    for row in mallocs:
        assignment = row.c.parent.parent
        if not assignment.isentity("BinaryOperator"):
            print(f"Malloc is not well formed ({row.c.location}). Skipping.")
            continue
        declref = row.c.parent.parent.children[0]
        if not declref.isentity("DeclRefExpr"):
            print(f"Malloc is not well formed ({row.c.location}). Skipping.")
            continue
        pointer_var = declref.unparse()
        size = row.c.args[0].unparse()
        assignment.instrument(Action.replace, code=f"hipHostMalloc((void**)&{pointer_var}, {size});")
        assignment.instrument(Action.remove_semicolon, verify=False)
        pinned_pointers.append(pointer_var)
    frees = ast.query('c{CallExpr}', where=lambda c: c.name == 'free')
    for row in frees:
        pointer_var = row.c.args[0].unparse()
        if pointer_var in pinned_pointers:
            row.c.instrument(Action.replace, code=f"hipFree({pointer_var});")
            row.c.instrument(Action.remove_semicolon, verify=False)

def introduce_shared_mem_buf_hip(kernel, kernel_wrapper, pointer_param, struct_map):
    # get pointer size and type
    pointer_var = pointer_param[1]
    pointer_type = pointer_param[0].spelling.replace("*", "").replace(" ", "").replace("__restrict", "")
    pointer_size = f"({pointer_var}_size_/sizeof({pointer_type}))" 
    # set dynamic shared memory size
    vds = kernel_wrapper.query('vd{VarDecl}', where=lambda vd: vd.name == 'dyn_shared')
    for row in vds:
        decl_string = row.vd.unparse()
        new_string = decl_string[:decl_string.index('=')+1] + f" {pointer_size}*sizeof({pointer_type});"
        row.vd.instrument(Action.replace, code=new_string)
        row.vd.instrument(Action.remove_semicolon, verify=False)
    # generate buffer fill loop string
    if pointer_type in struct_map:
        fill_string = ""
        for member in struct_map[pointer_type]:
            fill_string += f"{pointer_var}_cache[__idx__].{member[1]} = {pointer_var}[__idx__].{member[1]};\n"
    else:
        fill_string = f"{pointer_var}_cache[__idx__] = {pointer_var}[__idx__];\n"
    shared_mem_template = (f"extern __shared__ {pointer_type} {pointer_var}_cache[];\n"
                f"for (int __idx__ = threadIdx.x; __idx__ < {pointer_size}; __idx__ += blockDim.x)" + "{ \n"
                f"{fill_string}"
                "}\n__syncthreads();")
    # insert buffer fill loop string -- after condition check 
    bounds_check = kernel.query("cond{IfStmt} => ret{ReturnStmt}", where=lambda cond, ret: len(cond.children) == 2 and ret in cond.children)
    ## TODO: if multiple of these conditionals, find FIRST one (bounds check)
    if not bounds_check:
        print("Can't find top level bounds check. Returning. ")
        ast.reset()
        return 
    bounds_check = bounds_check[0].cond
    bounds_check.instrument(Action.after, code=";\n"+shared_mem_template)
    # replace refs to old pointer with new shared version 
    refs = kernel.query("ref{DeclRefExpr}", where=lambda ref: ref.name == pointer_var)
    for row in refs:
        row.ref.instrument(Action.replace, code=f"{pointer_var}_cache")

def introduce_shared_mem(ast, kernel_fn, wrapper_fn, report, struct_map, max_size=10000):
    params = [el for el in report if el != 'summary' and 'R' in report[el]['rw'] and report[el]['size'] < max_size]
    for param in params:
        kernel_param = [p for p in kernel_fn.signature.params if p[1] == param][0]
        introduce_shared_mem_buf_hip(kernel_fn, wrapper_fn, kernel_param, struct_map)
        ast.commit()
    ast.sync(commit=True)

def change_blocksize(ast, bs, bs_var="_blocksize_"):
    decls = ast.query('vd{VarDecl}', where=lambda vd: vd.name == bs_var)
    if not decls:
       print(f"Can't find blocksize var decl for {bs_var}.")
       return
    decl = decls[0].vd
    decl.instrument(Action.replace, code=f"const int {bs_var} = {bs}")

def time_kernel_bs_DSE(hip_ast, kernel_fn, device=None):
    all_times = {}
    blocksize = 16
    while blocksize <= 1024:
        change_blocksize(hip_ast, blocksize)
        hip_ast.sync(commit=True)
        timer = HIPKernelTimer(hip_ast)
        e2e_times, compute_times = timer.run(kernel_fn, num_runs=5, device=device)
        print(f"Blocksize = {blocksize}\n", "E2E Times:", e2e_times, "Compute Times:", compute_times)
        all_times[blocksize] = {}
        all_times[blocksize]['e2e'] = e2e_times
        all_times[blocksize]['compute'] = compute_times
        blocksize *= 2
    min_time = 1000000000000
    best_bs = None
    for bs in all_times:
        ave = np.mean(all_times[bs]['e2e'])
        print(f"Blocksize: {bs}, Average: {ave}")
        if ave < min_time:
            min_time = ave
            best_bs = bs
    min_time = 1000000000000
    best_bs = None
    for bs in all_times:
        ave = np.mean(all_times[bs]['compute'])
        print(f"Blocksize: {bs}, Average: {ave}\n ({all_times[bs]['compute']})")
        if ave < min_time:
            min_time = ave
            best_bs = bs
    change_blocksize(hip_ast, best_bs)
    hip_ast.sync(commit=True)


class HIPKernelTimer:
    def __init__(self, ast):
        # clone the ast for instrumentation and execution
        self.ast = ast.clone('debug-kernel-timer/', changes=True) 
        self.data = {}

    # returns profiling data
    def run(self, kernel_fn, num_runs=1, debug=False, device=None):

        # find kernel launch function call - hip macro instantiation
        kernel_launch = self.ast.query('src{Module} => mi{MacroInstantiation}', where=lambda mi: mi.unparse().startswith("hipLaunchKernelGGL"))
        if len(kernel_launch) == 0:
            print("Can't find kernel launch function to time. Returning.")
            return
        src1 = kernel_launch[0].src
        kernel_launch = kernel_launch[0].mi
        
        # find kernel wrapper function call 
        kernel_wrapper = self.ast.query(select='src{Module} => call{CallExpr}', where=lambda call: call.name == f'{kernel_fn}_wrapper_')
        if len(kernel_wrapper) == 0:
            print("Can't find kernel wrapper function to time. Returning.")
            return

        src2 = kernel_wrapper[0].src
        kernel_call = kernel_wrapper[0].call
        srcs = set([src1, src2])
        
        # include header <meta_cl> on sources
        for src in srcs:
            incl_artisan(src)

        # instrument wrapper function to add a timer
        kernel_call.instrument(Action.before, code='{ Timer __timer_e2e__([](double t) { Report::write("(\'e2e-timer\', %1%),", t); }, true);')
        kernel_call.instrument(Action.after, code=';}')

        # instrument launch call to add a timer and synchronise mechanisms 
        kernel_launch.instrument(Action.before, code='hipDeviceSynchronize();\n{ Timer __compute_timer__([](double t) { Report::write("(\'compute-timer\', %1%),", t); }, true);\n')
        kernel_launch.instrument(Action.after, code=';\nhipDeviceSynchronize();\n}')

        # step 3: query main function
        results2 = self.ast.query("fn{FunctionDecl}", where=lambda fn: fn.name == 'main')
        for res in results2:
            res.fn.instrument(Action.begin,
                              code='Report::connect(); Report::write("["); int ret = [] (auto argc, auto argv) { ')
            res.fn.instrument(Action.end,
                              code='  }(argc, argv);'
                                   'Report::write("]");\nReport::emit();\nReport::disconnect();'
                                    'return ret;')

        # sync AST to execute
        self.ast.sync(commit=True)
        if device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
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
            for el in self.data:
                if el[0] == 'compute-timer':
                    compute_time += el[1]
                if el[0] == 'e2e-timer':
                    e2e_time += el[1]
            compute_times.append(compute_time)
            e2e_times.append(e2e_time)

        return e2e_times, compute_times