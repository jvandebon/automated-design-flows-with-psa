from .instrumentation import CodeInstrument
from functools import partial
import subprocess

def pretty_print(code:str):
   p = subprocess.Popen(["clang-format", r"-style={BasedOnStyle: Google, BreakBeforeBraces: Allman, SortIncludes: false, BreakStringLiterals: false, ColumnLimit: 5000, AllowShortFunctionsOnASingleLine: false, AllowShortBlocksOnASingleLine: false}"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE, encoding='utf8')

   stdout, stderr = p.communicate(input=code)

   if stderr:
       raise RuntimeError("cannot pretty print: ", stderr)

   return stdout

def unparse_token_instr(instrument:CodeInstrument, token):
    annotation = instrument.get_annotations(token)
    if annotation:
        if CodeInstrument.IGNORE in annotation:
            return None
        elif CodeInstrument.REPLACE in annotation:
            return annotation[CodeInstrument.REPLACE]

        code = ""
        if CodeInstrument.BEFORE in annotation:
            code = annotation[CodeInstrument.BEFORE]
        code += token.spelling
        if CodeInstrument.AFTER in annotation:
            code += annotation[CodeInstrument.AFTER]
        return code
    else:
        return token.spelling


def unparse_token(token):
    return token.spelling

def unparse(cnode, instrument:CodeInstrument=None):
    offset_line = -1
    offset_col = -1

    code = ""

    if instrument is None:
        unparse_token_fn = unparse_token
    else:
        unparse_token_fn = partial(unparse_token_instr, instrument)

    tokens = [ t for t in cnode.__cursor__.get_tokens()]

    if len(tokens) == 0:
        return code

    if offset_line == -1:
        token_loc = tokens[0].extent
        offset_line = token_loc.start.line
        offset_col = token_loc.start.column


    for t in tokens:
        loc = t.extent

        if loc.start.line != offset_line:
            if loc.start.line > offset_line:
               code += "\n"*(loc.start.line - offset_line)
               offset_line = loc.start.line
               offset_col = 1

            else:
               raise RuntimeError("line out of sync")
        if loc.start.column != offset_col:
            if loc.start.column > offset_col:
               code += " "*(loc.start.column - offset_col)
               offset_col = loc.start.column
            else:
               raise RuntimeError("column out of sync")

        token_str = unparse_token_fn(t)
        if token_str is not None:
           code += token_str
        offset_line = loc.end.line
        offset_col = loc.end.column
    return code
