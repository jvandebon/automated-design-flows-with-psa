
from itertools import takewhile
import shutil

import os.path as osp
import difflib as dl

import re

def copy_tree(src, target, overwrite=True):

    # target cannot be inside source
    base0 = osp.abspath(src)
    base1 = osp.abspath(osp.dirname(target))

    if osp.isdir(target):
        if overwrite:
            shutil.rmtree(target)

    # removes infine recursion...
    def ignore(d, files):
        ignore_lst = []
        for f in files:
            absf = osp.join(d, f)
            if osp.isdir(absf) and osp.commonpath([absf]) == osp.commonpath([absf, target]):
                ignore_lst.append(f)
        return ignore_lst
    shutil.copytree(src, target, ignore=ignore)

# maps lines of original to new code
class SourceDiff:
    def __init__(self, original:str, transformed:str):
        self.diffs = []
        lines = dl.unified_diff(original.splitlines(), transformed.splitlines(), n=0)
        # checks for unified diff
        for line in lines:
            z = re.match(r"@@ \-(\d+),?(\d+)? \+(\d+),?(\d+)? @@", line)
            if z:
               g = z.groups()
               t = (int(g[0]),
                    1 if g[1] is None else int(g[1]),
                    int(g[2]),
                    1 if g[3] is None else int(g[3]))
               for n in range(0, t[1]):
                   self.diffs.append((t[0]+n, None))
               if t[1] == 0:
                   n = 0
               sp = t[0]+n + 1
               offset = t[2] + t[3] + (1 if t[3] == 0 else 0)
               self.diffs.append((sp, offset - sp))

    def map(self, line):
        # we need to find a directive that matches
        # our line.

        directive = None
        for d in self.diffs:
           if line == d[0]:
               directive = d
               break
           elif line > d[0]:
               directive = d # but keep looking
           else:
               break

        if directive is not None:
            if directive[1] is None:
                return None
            else:
               line = line + directive[1]

        return line




