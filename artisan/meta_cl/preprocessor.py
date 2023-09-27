import pcpp.preprocessor as prep
import io

### control
# 0: disabled
# 1: source (local) #defines
# 2: process #if* conditionals    
# 3: process #includes (global #defines)

class MetaCLPreprocessor(prep.Preprocessor):
    def __init__(self, control):       

        self.directives_to_ignore = set(['ifdef', 'ifndef', 'if', 'elif', 'else', 'endif', 'include', 'define', 'undef'])
        if control == 1:
            self.directives_to_ignore = self.directives_to_ignore - set(['define', 'undef'])
        elif control == 2:
            self.directives_to_ignore = set(['include'])
        elif control != 0:
            self.directives_to_ignore = set()
        
        super(MetaCLPreprocessor, self).__init__()


    def on_directive_handle(self,directive,toks,ifpassthru,precedingtoks):
        if directive.value in self.directives_to_ignore:
            raise prep.OutputDirective(prep.Action.IgnoreAndPassThrough)

        super(MetaCLPreprocessor, self).on_directive_handle(directive,toks,ifpassthru,precedingtoks)
        return None

    def on_comment(self,tok):
        return True

    def on_include_not_found(self,is_malformed,is_system_include,curdir,includepath):
        raise prep.OutputDirective(prep.Action.IgnoreAndPassThrough)

def preprocess(code, *, defs, args, control):
    if control is None:
        raise RuntimeError(f"invalid preprocessor control: '{control}'!")

    if type(control) == bool:
        if control:
            control = 3
        else:
            control = 0

    preprocessor = MetaCLPreprocessor(control)

    # specify defs
    if len(defs) > 0 and (control > 0):
        iargs = iter(defs)
        for arg in iargs:
            if arg == '-D':
                preprocessor.define(next(iargs).replace("=", " ", 1))
            elif arg == '-U':
                preprocessor.undef(next(iargs))

    # specify include paths
    if len(args) > 0 and (control > 1):
        iargs = iter(args)
        for arg in iargs:
            if arg == '-I':
                path = next(iargs)
                preprocessor.add_path(path)


    preprocessor.parse(input=code)
    outstr = io.StringIO()
    preprocessor.write(oh=outstr)
    preproc_code = outstr.getvalue()
    outstr.close()

    return preproc_code


