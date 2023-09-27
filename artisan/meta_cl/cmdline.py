import pathlib
import sys
import argparse
import os
import shlex

supported_source_ext = {'.cpp', '.cc', '.c', '.cl', '.cu', '.h', '.hpp', '.hh' }

argp = argparse.ArgumentParser(add_help=False)
argp.add_argument('-D', dest = 'defines',  nargs = 1, action = 'append', help = 'Predefine name as a macro [with value]')
argp.add_argument('-U', dest = 'undefines',  nargs = 1, action = 'append', help = 'Pre-undefine name as a macro')
argp.add_argument('-I', dest = 'includes',  nargs = 1, action = 'append', help = "Path to search for unfound #include's")
argp.add_argument('-include', dest = 'headers',  nargs = 1, action = 'append', help = "Include headers")

def parse(cmd=None):
    # we are keeping it simple. Sources correspond to
    # args that do not start with '-' and end with one
    # of the supported extensions (supported_source_ext)

    if cmd is None:
        cmdlst = sys.argv[1:]
    else:
        cmdlst = shlex.split(cmd)

    extra_flags = ""
    # add META_CL_CXXFLAGS if exists
    cxxflags = os.getenv('META_CL_CXXFLAGS')
    if cxxflags is not None:
        extra_flags = extra_flags + " " + cxxflags

    # add META_CL_CXXFLAGS_FE if exists
    cxxflags_fd = os.getenv('META_CL_CXXFLAGS_FE')
    if cxxflags_fd is not None:
        extra_flags = extra_flags + " " + cxxflags_fd

    cmdlst.extend(shlex.split(extra_flags))

    # parse arguments
    spec, unspec = argp.parse_known_args(cmdlst)

    source_names = []
    compiler_args = []
    preproc_args = []

    for elem in unspec:
        if elem[0] != '-' and pathlib.Path(elem).suffix in supported_source_ext:
            source_names.append(elem)
        else:
            compiler_args.append(elem)


    if spec.includes is not None:
        for incl in spec.includes:
           compiler_args.extend(['-I', incl[0]])

    if spec.headers is not None:
        for s in spec.headers:
           compiler_args.extend(['-include', s[0]])

    if spec.defines is not None:
        for s in spec.defines:
           preproc_args.extend(['-D', s[0]])

    if spec.undefines is not None:
        for s in spec.undefines:
           preproc_args.extend(['-U', s[0]])

    return (source_names, compiler_args, preproc_args)


















