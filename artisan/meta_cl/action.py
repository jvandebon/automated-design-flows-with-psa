from .utils import token_id
from .instrumentation import CodeInstrument
import clang.cindex as ci

class Action:
    @staticmethod
    def before(instr, cnode, tokens, code):
        instr.annotate_token(cnode, tokens[0], CodeInstrument.BEFORE, code)

    @staticmethod
    def after(instr, cnode, tokens, code):
        instr.annotate_token(cnode, tokens[-1], CodeInstrument.AFTER, code)

    @staticmethod
    def replace(instr, cnode, tokens, code):
        instr.annotate_token(cnode, tokens[0], CodeInstrument.REPLACE, code)
        for token in tokens[1:]:
            instr.annotate_token(cnode, token, CodeInstrument.IGNORE)

    @staticmethod
    def begin(instr, cnode, tokens, code):
        found_token = False
        for token in tokens:
            if token.kind == ci.TokenKind.PUNCTUATION and token.spelling == '{':
                found_token = True
                break
        if found_token:
            # we add "\n" to allow freezing the node
            instr.annotate_token(cnode, token, CodeInstrument.AFTER, code)
        else:
            raise RuntimeError("cannot find begin ({)!")

    @staticmethod
    def end(instr, cnode, tokens, code):

        found_token = False
        for token in reversed(tokens):
            if token.kind == ci.TokenKind.PUNCTUATION and token.spelling == '}':
                found_token = True
                break
        if found_token:
            instr.annotate_token(cnode, token, CodeInstrument.BEFORE, code)
        else:
            raise RuntimeError("cannot find end (})!")

    @staticmethod
    def remove_semicolon(instr, cnode, tokens, verify=True):

        last_token = tokens[-1]

        module_tokens = cnode.module.tokens # get all the module tokens

        found = False
        for mt in module_tokens:
            if found:
                # next token must be a semicolon
                if (mt.kind == ci.TokenKind.PUNCTUATION) and (mt.spelling == ';'):
                    instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)
                else:
                    if verify:
                        raise RuntimeError("cannot find semicolon!")
                break
            else:
                if token_id(mt) == token_id(last_token):
                    found = True

    @staticmethod
    def pragmas(instr, cnode, tokens, fn):

        pragmas = cnode.pragmas
        if pragmas is None or len(pragmas) == 0:
            return

        module = cnode.module
        if module is None:
            module = cnode

        entry_tokens = {token_id(ini):(code, end) for (code, ini, end) in pragmas }
        exit_tokens = set([token_id(end) for (_, _, end) in pragmas ])

        module_tokens = cnode.module.tokens # get all the module tokens

        ignore_pragma = False

        for mt in module_tokens:
            token_mt = token_id(mt)

            if token_mt in entry_tokens:
                 ret = fn(entry_tokens[token_mt][0])
                 if type(ret) == bool and not ret:
                     instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)
                     ignore_pragma=True
                 elif type(ret) == str:
                     instr.annotate_token(cnode, mt, CodeInstrument.REPLACE, code=ret)
                     ignore_pragma=True

            elif token_mt in exit_tokens:
                if ignore_pragma:
                    instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)
                    ignore_pragma = False
            else:
                if ignore_pragma:
                    instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)


    @staticmethod
    def attributes(instr, cnode, tokens, fn):

        attribs = cnode.attributes
        if attribs is None or len(attribs) == 0:
            return

        module = cnode.module

        entry_tokens = {token_id(attr[1]):attr[0] for attr in attribs }
        exit_tokens = {token_id(attr[2]) for attr in attribs }

        module_tokens = cnode.module.tokens # get all the module tokens

        ignore_attribute = False

        for mt in module_tokens:
            token_mt = token_id(mt)

            if token_mt in entry_tokens:
                 ret = fn(entry_tokens[token_mt])
                 if type(ret) == bool and not ret:
                     instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)
                     ignore_attribute=True
                 elif type(ret) == str:
                     instr.annotate_token(cnode, mt, CodeInstrument.REPLACE, code=ret)
                     ignore_attribute=True

            elif token_mt in exit_tokens:
                if ignore_attribute:
                    instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)
                    ignore_attribute = False
            else:
                if ignore_attribute:
                    instr.annotate_token(cnode, mt, CodeInstrument.IGNORE)














