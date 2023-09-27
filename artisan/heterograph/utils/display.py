from pygments import highlight
from pygments.lexers.c_cpp import CppLexer
from pygments.formatters import TerminalTrueColorFormatter

def terminal_cpp(code):
    return highlight(code, CppLexer(), TerminalTrueColorFormatter())

