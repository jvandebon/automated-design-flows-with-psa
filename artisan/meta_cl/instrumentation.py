from .utils import token_id
import clang.cindex as ci

class CodeInstrument:
    # token annotations
    BEFORE=0
    AFTER=1
    IGNORE=2
    REPLACE=3

    def __init__(self):
        # we use clang's translation unit since we work at token level which is not exposed from cnode

        # tu => { token id: { BEFORE/AFTER/IGNORE/REPLACE: <code> } }
        self.annotations = { }

        # sources changed
        self.__changed = set()

    def clone(self):
        instr = CodeInstrument()
        instr.annotations = self.annotations
        instr.__changed = self.__changed

    @property
    def num_changes(self):
        sum = 0
        for tu in self.annotations:
            sum = sum + self.num_module_changes(tu)

        return sum

    def num_module_changes(self, tu):
        return len(self.annotations[tu]) if tu in self.annotations else 0

    @property
    def changed(self):
        return self.__changed

    def reset(self):
        self.annotations.clear()
        self.__changed.clear()

    def get_annotations(self, token):
        tu = token.cursor.translation_unit

        if tu not in self.annotations:
            return None
        tu_record = self.annotations[tu]

        tid = token_id(token)

        if tid not in tu_record:
            return None

        return tu_record[tid]

    def annotate_token(self, cnode, token, annotation, code=""):
        tu = cnode.__cursor__.translation_unit

        if tu not in self.annotations:
            self.annotations[tu] = {}

        self.__changed.add(cnode.srcname)

        tu_record = self.annotations[tu]

        tid = token_id(token)

        if tid not in tu_record:
            tu_record[tid] = {}

        token_record = tu_record[tid]

        if annotation in token_record:
            raise RuntimeError("cnode %d already instrumented!" % cnode.id)
        token_record[annotation] = code

        # validation check
        if len(token_record) > 1 and sum(token_record.keys()) > 1:
            raise RuntimeError('conflited instrumentation instructions')

    def apply(self, cnode, action, **kwargs):
        # format:
            # action(instr, cnode, tokens)
            tokens = list(cnode.__cursor__.get_tokens())
            action(instr=self, cnode=cnode, tokens=tokens, **kwargs)


    '''
    def register(self, cnode, code=None, before=False, around=False, after=False, begin=False, end=False, prologue=None, epilogue=None, prefix='', suffix='\n'):

        # ensure that only one position is specified
        s = sum([1 if p else 0 for p in [before,around,after,begin,end,prologue,epilogue]])

        if s != 1:
            raise RuntimeError("please specify only one argument: 'begin', 'around', 'after', 'begin', 'end', 'prologue', or 'epilogue'!")

        if (not prologue) and (not epilogue) and (code is None):
            raise RuntimeError("must specify code!")

        if (prologue or epilogue) and (code is not None):
            raise RuntimeError("code cannot be specified when 'prologue' or 'epilogue' is specified!")


        tokens = list(cnode.__cursor__.get_tokens())

        if len(tokens) == 0:
            raise RuntimeError(f"cnode '{cnode.entity}' in '{cnode.location}' does not have tokens associated!")

        if before:
            self.register_token(tu, cnode, tokens[0], CodeInstrument.BEFORE, code, prefix, suffix)

        elif after:
            self.register_token(tu, cnode, tokens[-1], CodeInstrument.AFTER, code, prefix, suffix)

        elif around:
            self.register_token(tu, cnode, tokens[0], CodeInstrument.REPLACE, code, prefix, suffix)
            for token in tokens[1:]:
                self.register_token(tu, cnode, token, CodeInstrument.IGNORE, code, prefix, suffix)

        elif begin:
            found_token = False
            for token in tokens:
                if token.kind == ci.TokenKind.PUNCTUATION and token.spelling == '{':
                    found_token = True
                    break
            if found_token:
                # we add "\n" to allow freezing the node
                self.register_token(tu, cnode, token, CodeInstrument.AFTER, "\n" + code, prefix, suffix)
            else:
                raise RuntimeError("cannot find begin ({)!")
        elif end:
            found_token = False
            for token in reversed(tokens):
                if token.kind == ci.TokenKind.PUNCTUATION and token.spelling == '}':
                    found_token = True
                    break
            if found_token:
                self.register_token(tu, cnode, token, CodeInstrument.BEFORE, code + "\n", prefix, suffix)
            else:
                raise RuntimeError("cannot find end (})!")
        elif prologue:
            pass

        elif epilogue:
            # start with last token: tokens[-1]
            lastt = tokens[-1]
            module_tokens = cnode.module.tokens # get all the module tokens

            # find the token
            idx = 0
            found = False
            for mt in module_tokens:
                if (token_id(mt) == token_id(lastt)):
                    found = True
                    break
                idx = idx + 1

            module_tokens = module_tokens[idx:]

            pos = 1
            for mt in module_tokens:
                ret = epilogue(mt, pos)
                if type(ret) == bool:
                    if not ret:
                        break
                else:
                    self.register_token(tu, cnode, mt, CodeInstrument.REPLACE, code=ret)
                pos = pos + 1
        else:
            raise RuntimeError("internal error: nothing registered!")
    '''















