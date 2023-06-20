from meta_cl import *
from profilers import *
import numpy as np

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