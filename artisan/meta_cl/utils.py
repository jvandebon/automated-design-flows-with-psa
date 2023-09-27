import clang.cindex as ci
import os.path as osp

# expose extra clang functions
funcs = [
         ('clang_Location_isFromMainFile',[ci.SourceLocation], ci.c_int),
         ('clang_getRemappings',[ci.c_interop_string], ci.c_object_p),
         ('clang_remap_getNumFiles', [ci.c_object_p], ci.c_int),
        ]
for f in funcs:
   ci.register_function(ci.conf.lib, f, ignore_errors=False)

def clang_location_str(loc:ci.SourceLocation, workdir=None):
    filepath = "" if loc.file is None else loc.file.name

    if filepath != "":
        if workdir is not None:
            if not osp.isabs(filepath):
                filepath = osp.join(workdir, filepath)

    return f"{filepath}:{loc.line}:{loc.column}"

def token_id(token):
   return token.int_data[1]

def shape_type(type):
    # returns:
    #     => (base type, None): scalar
    #     => (base type, [10]): array of 10 elements, etc. 0 is undefined

    class ShapeType:
      def __init__(self, base_type, dim):
          self.base_type = base_type
          self.dim = dim

      def __repr__(self):
         return f"<{self.base_type.spelling}> {self.dim}"

    if type is None:
        return None

    if type.kind != ci.TypeKind.CONSTANTARRAY and type.kind != ci.TypeKind.VARIABLEARRAY:
        return ShapeType(type, None)

    size = []
    _type = type
    while _type.kind == ci.TypeKind.CONSTANTARRAY or _type.kind == ci.TypeKind.VARIABLEARRAY:
        dm = _type.get_array_size()
        if dm != -1:
            size.append(dm)
        else:
            arr = _type.spelling
            p0 = arr.index('[')
            p1 = arr.index(']')
            size.append(arr[p0+1:p1])
        _type = _type.get_array_element_type()
    return ShapeType(_type, size)

def get_basetype_from_typedef(t):
    type_decl = ci.conf.lib.clang_getTypeDeclaration(t)
    base_type = ci.conf.lib.clang_getTypedefDeclUnderlyingType(type_decl)
    return base_type


def get_cpp_attribute(cnode, name):

    ret = None
    attributes = cnode.attributes
    if attributes is not None:
        for attrs in attributes:
            if name in attrs[0]:
                if ret is None:
                    ret = []
                ret.append(attrs[0][name])

    return ret







