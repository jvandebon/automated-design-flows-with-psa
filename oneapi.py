from meta_cl import *
from util import *

def device_queue_and_exception_handler():
    return """static auto exception_handler = [](sycl::exception_list e_list) {
            for (std::exception_ptr const &e : e_list) {
                try { 
                    std::rethrow_exception(e); 
                } catch (std::exception const &e) {
                    std::terminate();
                }
            }
        };
        queue getQueue(){
        #ifdef FPGA 
            ext::intel::fpga_selector device_selector;
        #else 
            #ifdef FPGA_EMULATOR
                ext::intel::fpga_emulator_selector device_selector;
            #else
                cpu_selector device_selector;
            #endif
        #endif

            return queue(device_selector, exception_handler);
        }\n"""

def acc_mode(rw):
    mode = {'RW': "read_write", 'W':  "write", 'R': "read"}
    if not rw in mode:
        rw = 'RW'
    return mode[rw]

#TODO: NOT ROBUST
def type_of(param):
    return param.type.spelling[:param.type.spelling.index("*")].strip()

def sycl_housekeeping(fn, kernel_name):
    # necessary sycl include stmts
    housekeeping = "#include <CL/sycl.hpp>\n"
    housekeeping += "#include <chrono>\n"
    housekeeping  += "#if FPGA || FPGA_EMULATOR\n#include <sycl/ext/intel/fpga_extensions.hpp>\n#endif\n"
    housekeeping  += "using namespace cl::sycl;\n"
    housekeeping += device_queue_and_exception_handler()
    
    # declare kernel template
    housekeeping += "class %s;\n" % kernel_name

    # instrument module with housekeeping code
    fn.module.instrument(Action.before, code=housekeeping)

def setup_device_queue(calls):
    # add queue to fn 
    fn_scope = calls[0].fndecl.body
    fn_scope.instrument(Action.begin, code='queue q = getQueue();\n')
    
    for row in calls:
        # add queue arg
        new_call = "%s(%s);\n" % (row.c.name, ','.join([c.unparse() for c in row.c.args] + ['q']))
        row.c.instrument(Action.replace, code=new_call)
        row.c.instrument(Action.remove_semicolon)

def loop_kernel_calls_for_warmup(ast, kernel_name):

    # find call to kernel
    kernel_call = ast.query('src{Module} => c{CallExpr}', where=lambda c: c.name == kernel_name)
    if not kernel_call:
        return
    src = kernel_call[0].src
    kernel_call = kernel_call[0].c
    
    # add macros to #define NUM_RUNS and include timing headers if needed
    macros = "#define NUM_RUNS 2\n"
    src.instrument(Action.before, code=macros)

    # wrap call in a loop
    call_loop = "for (size_t RUN = 0; RUN < NUM_RUNS; RUN++) {\n %s; }\n" % kernel_call.unparse().strip()
    kernel_call.instrument(Action.replace, code=call_loop)


def fn_to_oneapi_kernel(fn, kernel_name, restrict):
    # for all pointer params - buffer decl, in kernel access (R/W), set_final_data (W)
    params = fn.query('p{ParmDecl}')
    param_names = [row.p.name for row in params]
    buffer_decls = ""
    kernel_accessors = ""
    final_data = ""
    lambda_args = []
    for row in params:
        param = row.p
        if pointer_type(param.type):
            size = "%s_size_" % param.name
            if not size in param_names:
                print("Can't determine size of pointer: %s, exiting..." % param)
                exit(1)
            rw = read_or_write_param(param, fn)
            buffer_decls += "buffer<%s> %s_buf(%s, range<1>(%s/sizeof(%s)));\n" % (type_of(param), param.name, param.name, size, type_of(param))
            kernel_accessors += "auto %s = %s_buf.get_access<access::mode::%s>(cgh);\n" % (param.name, param.name, acc_mode(rw))
            if 'W' in rw:
                final_data += "%s_buf.set_final_data(%s);\n" % (param.name, param.name)
        lambda_args.append(param.name)

    # wrap body in cgh.single_task
    directives = ""
    if restrict:
        directives += "[[intel::kernel_args_restrict]]"
    kernel_lambda = "cgh.single_task<class %s>([=]() %s %s);\n" % (kernel_name, directives, fn.body.unparse())

    # wrap kernel lambda in auto evt = ...; evt.wait();
    kernel_event = "/*Timer Start*/\nauto evt = q.submit([&](handler& cgh) {\n%s%s});\nevt.wait();/*Timer End*/\n" % (kernel_accessors, kernel_lambda)
    
    # put all kernel pieces together
    kernel_fn_body = "%s%s%s" % (buffer_decls, kernel_event, final_data)

    # add queue argument to function
    params = ["%s %s" % (p[0].spelling, p[1]) for p in fn.signature.params] + ["queue q"]
    sig = "%s %s(%s)" % (fn.signature.rettype.spelling, fn.name, ','.join(params))

    # replace function with new kernel version
    fn.instrument(Action.replace, code="%s{\n%s\n}" % (sig, kernel_fn_body))

def replace_var_refs(node, old, new):
    refs = node.query('r{DeclRefExpr}', where=lambda r: r.unparse() == old)
    for row in refs:
        row.r.instrument(Action.replace, code=new)


def fn_to_oneapi_kernel_explicitmem(fn, kernel_name, restrict):

    # for all pointer params - allocate device mem, access device pointer in kernel, set up copy events
    params = fn.query('p{ParmDecl}')
    param_names = [row.p.name for row in params]
    pointer_params = [row.p for row in params if pointer_type(row.p.type)]

    device_mallocs = ""
    kernel_pointers = ""
    copy_to_device = ""
    copy_to_host = ""
    pointer_cleanup = ""
    to_replace = []

    for param in pointer_params:
        size = "%s_size_" % param.name
        if not size in param_names:
            print("Can't determine size of pointer: %s, exiting..." % param.name)
            exit(1)
        rw = read_or_write_param(param, fn)
        base_type = type_of(param)
        device_mallocs += "%s *%s_ptr = malloc_device<%s>(%s/sizeof(%s),q);\n" % (base_type, param.name, base_type, size, base_type)
        device_mallocs += "if(%s_ptr == nullptr) { std::cerr << \"Failed to allocate space for %s.\\n\"; return; }\n" % (param.name, param.name)
        pointer_cleanup += "sycl::free(%s_ptr,q);\n" % param.name
        kernel_pointers += "device_ptr<%s> %s_ptrd(%s_ptr);\n" % (base_type, param.name, param.name)
        to_replace.append((param.name, param.name+"_ptrd"))
        if 'R' in rw:
            copy_to_device += "auto %s_to_device = q.memcpy(%s_ptr, %s, %s);\n" % (param.name, param.name, param.name, size)
            copy_to_device += "%s_to_device.wait();\n" % param.name
        if 'W' in rw:
            copy_to_host += "auto %s_to_host = q.memcpy(%s, %s_ptr, %s);\n" % (param.name, param.name, param.name, size)
            copy_to_host += "%s_to_host.wait();\n" % param.name

    # wrap body in cgh.single_task
    directives = ""
    if restrict:
        directives += "[[intel::kernel_args_restrict]]"
    kernel_lambda = "cgh.single_task<class %s>([=]() %s { %s %s});\n" % (kernel_name, directives, kernel_pointers, fn.body.unparse()[1:-1])

    # wrap kernel lambda in auto evt = ...; evt.wait();
    kernel_event = "auto evt = q.submit([&](handler& cgh) {\n%s});\nevt.wait();\n" % (kernel_lambda)
    
    # put all kernel pieces together
    kernel_fn_body = "%s%s%s%s%s" % (device_mallocs, copy_to_device, kernel_event, copy_to_host, pointer_cleanup)

    # add queue argument to function
    params = ["%s %s" % (p[0].spelling, p[1]) for p in fn.signature.params] + ["queue q"]
    sig = "%s %s(%s)" % (fn.signature.rettype.spelling, fn.name, ','.join(params))

    # replace function with new kernel version
    fn.instrument(Action.replace, code="%s{\n%s\n}" % (sig, kernel_fn_body))
    
    # replace old pointer references to new device pointers (can't instrument all at once, sync first)
    ast = fn.ast
    ast.commit()
    kernel = ast.query('f{FunctionDecl} => evt{LambdaExpr} => kernel{LambdaExpr}', where=lambda f: f.name == fn.name)[0].kernel
    for i,j in to_replace:
        replace_var_refs(kernel, i, j)


def usm_malloc_free(base_type, name, size, host_or_device):
    malloc = "%s *%s_ptr = malloc_%s<%s>(%s/sizeof(%s),q);\n" % (base_type, name, host_or_device, base_type, size, base_type)
    malloc += "if(%s_ptr == nullptr) { std::cerr << \"Failed to allocate space for %s.\\n\"; return; }\n" % (name, name)
    free = "sycl::free(%s_ptr,q);\n" % name 
    return malloc, free

def basic_kernel_to_zerocopy(ast, kernel_fn):

    data = {}

    # find buffers, get information, remove 
    buffers = kernel_fn.query('stmt{DeclStmt} => d{VarDecl} => c1{CallExpr} => c2{CallExpr} ', where=lambda d,c1,c2: d.unparse()[:6] == c1.name == 'buffer' and c2.name =='range')
    for row in buffers:
        tmp = row.d.unparse()
        buf_type = tmp[tmp.find('<')+1:tmp.find('>')]
        tmp = row.c1.unparse()
        buf_name = tmp[:tmp.find('(')]
        buf_ptr = tmp[tmp.find('(')+1:tmp.find(',')]
        tmp = row.c2.unparse()
        buf_size = tmp[tmp.find('('):tmp.rfind(')')+1]
        data[buf_name] = {'ptr': buf_ptr, 'size': buf_size, 'type': buf_type}
        data[buf_name]['buf_stmt'] = row.stmt

    # find accessors for buffers, get information, remove
    accessors = kernel_fn.query('stmt{DeclStmt} ={1}> d{VarDecl} ={1}> c{CallExpr} => m{MemberRefExpr}', where=lambda c: c.name =='get_access')
    for row in accessors:
        acc_name = row.d.name
        buffer = row.m.children[0].name
        acc_mode = row.m.children[1].name
        data[buffer]['accessor'] = acc_name
        data[buffer]['rw'] = ''
        if 'read' in acc_mode:
            data[buffer]['rw'] += 'R'
        if 'write' in acc_mode:
            data[buffer]['rw'] += 'W'
        data[buffer]['acc_stmt'] = row.stmt

    copy_to_host = ""
    usm_cleanup = ""
    usm_ptrs = ""
    to_replace = []

    for buffer in data:
        info = data[buffer]
        
        usm_mallocs = ""
        copy_events = ""
        copy_to_usm = ""

        # if device pointers: malloc device, copy over
        if info['rw'] == "RW":
            malloc, free = usm_malloc_free(info['type'], info['ptr'], info['size'], "device")
            copy_events += "auto %s_to_device = q.memcpy(%s_ptr, %s, %s);\n%s_to_device.wait();\n" % (info['ptr'], info['ptr'], info['ptr'], info['size'], info['ptr'])
            usm_mallocs += malloc 
            usm_cleanup += free  
            usm_ptrs += "device_ptr<%s> %s__(%s_ptr);\n" % (info['type'], info['ptr'], info['ptr'])
            to_replace.append((info['accessor'], info['ptr']+"__"))
            copy_to_host += "memcpy(%s, %s_ptr, %s);\n" % (info['ptr'], info['ptr'], info['size'])       

        # if read only: malloc host, memcopy
        if info['rw'] == 'R':
            malloc, free = usm_malloc_free(info['type'], info['ptr'], info['size'], "host")
            copy_to_usm += "memcpy(%s_ptr, %s, %s);\n" % (info['ptr'], info['ptr'], info['size'])
            usm_mallocs += malloc 
            usm_cleanup += free   
            usm_ptrs += "host_ptr<%s> %s__(%s_ptr);\n" % (info['type'], info['ptr'], info['ptr'])
            to_replace.append((info['accessor'], info['ptr']+"__"))

        # if write only: malloc host 
        if info['rw'] == 'W':
            malloc, free = usm_malloc_free(info['type'], info['ptr'], info['size'], "host")
            copy_to_host += "memcpy(%s, %s_ptr, %s);\n" % (info['ptr'], info['ptr'], info['size'])
            usm_mallocs += malloc 
            usm_cleanup += free   
            usm_ptrs += "host_ptr<%s> %s__(%s_ptr);\n" % (info['type'], info['ptr'], info['ptr'])
            to_replace.append((info['accessor'], info['ptr']+"__")) 

        info['buf_stmt'].instrument(Action.replace, code=usm_mallocs+copy_events+copy_to_usm)
        info['acc_stmt'].instrument(Action.replace, code='')

    # declare usm pointers in kernel 
    kernel_lambdas = kernel_fn.query('evt{LambdaExpr} => kernel{LambdaExpr} ={1}> body{CompoundStmt}')[0]
    kernel_lambdas.body.instrument(Action.begin, code=usm_ptrs)

    # replace references to buffers in kernel
    kernel = kernel_lambdas.kernel
    for i,j in to_replace:
        replace_var_refs(kernel, i, j)

    # remove buffer copies
    buffer_copies = kernel_fn.query('c{CallExpr}', where=lambda c: c.name == 'set_final_data')
    for row in buffer_copies:
        row.c.instrument(Action.replace, code='')
        row.c.instrument(Action.remove_semicolon)

    # copy back to host and clean up after kernel 
    #TODO: assumes only one 'wait' call 
    kernel_wait = kernel_fn.query('c{CallExpr}', where=lambda c: c.name == 'wait')[0].c
    kernel_wait.instrument(Action.after, code=';\n'+copy_to_host+usm_cleanup)

    ast.sync(commit=True)


def map_to_oneapi_basic(ast, fn, restrict=False, kernel_name="Kernel"):

    # find call to fn (make sure only in a single function scope)
    calls = ast.query('fndecl{FunctionDecl} => c{CallExpr}', where=lambda c: c.name == fn.name)
    if not calls or len(set([row.fndecl for row in calls])) > 1:
        print("Calls to kernel from different functions, not supported.\n")
        return

    # sycl housekeeping code
    sycl_housekeeping(fn, kernel_name)

    # setup device queue
    setup_device_queue(calls)

    # transform kernel fn 
    fn_to_oneapi_kernel(fn, kernel_name, restrict)
    ast.commit()
    loop_kernel_calls_for_warmup(ast, kernel_name)
    ast.commit()

def map_to_oneapi_explicitmem(ast, fn, restrict=False, kernel_name="Kernel"):

    # find call to fn (make sure only in a single function scope)
    calls = ast.query('fndecl{FunctionDecl} => c{CallExpr}', where=lambda c: c.name == fn.name)
    if not calls or len(set([row.fndecl for row in calls])) > 1:
        print("Calls to kernel from different functions, not supported.\n")
        return

    # sycl housekeeping code
    sycl_housekeeping(fn, kernel_name)

    # setup device queue
    setup_device_queue(calls)

    # transform kernel fn 
    fn_to_oneapi_kernel_explicitmem(fn, kernel_name, restrict)
    ast.commit()
    loop_kernel_calls_for_warmup(ast, kernel_name)
    ast.commit()

def remove_pragma(pragma):
    return ""

def unroll_fixed_oneapi_loops(ast, scope, max_iters=100):
    loops = scope.query('l{ForStmt}', 
            where=lambda l: (   get_loop_info(l)['start'].isdigit() 
                                and get_loop_info(l)['end'].isdigit() 
                                and abs(int(get_loop_info(l)['end']) 
                                - int(get_loop_info(l)['start'])) < max_iters))
    for row in loops:
        unroll_loop(ast, row.l, remove=False)
    ast.commit()

def unroll_loop(ast, loop, factor="", remove=True):
    if remove and loop.pragmas:
        loop.instrument(Action.pragmas,fn=remove_pragma)
        loop, = ast.commit(track=[loop])
    loop.instrument(Action.before,code=f"#pragma unroll {factor}\n")
    return loop

def parse_unroll_pragmas(pragma):
    if pragma [0] == "unroll":
        return "loop"
    else:
        return False # exclude pragma

def unroll_until_overmap_dse(ast, kernel, target='a10'):
    outer_loop = kernel.query('l{ForStmt}', where=lambda l: l.is_outermost())[0].l

    gen_oneapi_report(ast)
    area_info = parse_reported_utilisation(f"{ast.workdir}/{target}_report.prj/reports")
    loop_info = parse_reported_loop_info(kernel,f"{ast.workdir}/{target}_report.prj/reports")

    percents = list(area_info.values())
    overmapped = [p > 100 for p in percents]
    ii = loop_info[outer_loop.id]['II']

    n=1
    while not any(overmapped) and ii == '1':
        n*=2
        ast.parse_pragmas(rules=[parse_unroll_pragmas])
        outer_loop = unroll_loop(ast, outer_loop,factor=n)
        kernel,outer_loop = ast.commit(track=[kernel,outer_loop])
        ast.sync()

        gen_oneapi_report(ast)
        area_info = parse_reported_utilisation(f"{ast.workdir}/{target}_report.prj/reports")
        loop_info = parse_reported_loop_info(kernel,f"{ast.workdir}/{target}_report.prj/reports")

        new_percents = list(area_info.values())
        if new_percents == percents:
            break

        percents = new_percents
        overmapped = [p > 100 for p in percents]
        ii = loop_info[outer_loop.id]['II']
        print('unroll by', n, 'ii='+ii, percents)

    # rollback 
    n/=2
    ast.parse_pragmas(rules=[parse_unroll_pragmas])
    print("DSE finished, rolling back to ", n)
    if int(n) == 1:
        outer_loop.instrument(Action.pragmas,fn=remove_pragma)
    else:
        outer_loop = unroll_loop(ast,outer_loop,factor=int(n))
    ast.sync(commit=True) 

## INTEL DESIGN REPORT PARSING

def gen_oneapi_report(oneapi_ast, target='a10'):
    oneapi_ast.exec(rule=f"{target}_report")

def get_area_report_json(report_path):
    with open(report_path+"/resources/json/area.json", 'r') as f:
        return json.load(f)  

def parse_reported_utilisation(report_path):
    area_report = get_area_report_json(report_path)
    resource_use = dict(zip(area_report['columns'][1:-1], area_report['total_percent']))
    return resource_use

def parse_reported_loop_info(scope, report_path):    
    # read in relevant json report files
    summary_data = get_loops_json_data(report_path)
    attr_data = get_loop_attr_json_data(report_path)
    # map report blocks to code locations
    loc_to_block = map_block_locations(summary_data)
    # find all loops contained within main loop, map each to a report blocks
    loops = [row.l for row in scope.query('l{loop}')]
    loops_to_blocks = map_loops_to_blocks(loops, loc_to_block)
    # extract relevant block information from json sources 
    block_summary_info = get_block_summary_info(summary_data)
    block_attr_info = get_block_attr_info(attr_data)
    # map all block information to corresponding loop
    loop_info = {}
    for l in loops_to_blocks:
        loop_info[l] = {}
        block = loops_to_blocks[l]
        if block in block_summary_info:
            loop_info[l].update(block_summary_info[block])
        if block in block_attr_info:
            loop_info[l].update(block_attr_info[block])

    return loop_info


def loc_str(loc):
    loc_str = loc['filename'][loc['filename'].rfind('/')+1:]
    loc_str += ":%d" % loc['line']
    return loc_str

def get_loop_attr_json_data(report_path):
    with open(report_path+"/resources/json/loop_attr.json", 'r') as f:
        return json.load(f)
    return {}

def get_loops_json_data(report_path):
    with open(report_path+"/resources/json/loops.json", 'r') as f:
        return json.load(f)
    return {}

def map_block_locations(summary_data):
    loc_to_block = {}
    blocks = summary_data['children']
    for block in blocks:
        loc = block['debug'][0][0]
        loc_to_block[loc_str(loc)] = block['name']
        for child in block['children']:
            blocks.append(child)
    return loc_to_block

def get_block_summary_info(summary_data):
    block_info = {}
    blocks = summary_data['children']
    for block in blocks:
        s = block['data'][2]
        serial = False
        for x in [d["text"] for d in block['details'] if d['type'] == "text"]:
            if "executed serially" in x:
                serial = True
                break
        block_info[block['name']] = {'S': s, 'serial': serial}
        for child in block['children']:
            blocks.append(child)
    return block_info

def determine_loop_block(loop, loc_to_block):
    loc = str(loop.location)
    # discard column
    loc = loc[:loc.rfind(":")]
    if loc in loc_to_block:
        return loc_to_block[loc]
    else:
        return None # no blocks for unrolled loops, eg.


def map_loops_to_blocks(loops, loc_to_block):
    loops_to_blocks = {}
    for l in loops:
        block = determine_loop_block(l, loc_to_block)
        if block:
            loops_to_blocks[l.id] = block
    return loops_to_blocks

def get_block_attr_info(attr_data):
    block_info = {}
    blocks = attr_data['nodes']
    for block in blocks:
        ii, lt = 0, 0
        if 'ii' in block:
            ii = block['ii']
        if 'lt' in block:
            lt = block['lt']
        block_info[block['name']] = {'II': ii, 'L': lt}
        if 'children' in block:
            for child in block['children']:
                blocks.append(child)
    return block_info



