from tabulate import tabulate
from colorama import Fore, Style
import copy

class QueryResultSet:

    def __init__(self, g,  qgraph, results:list):
        # result can be two formats:
        #     (1) a list of matches [ match0, match1, ... ], where
        #         each match is {'id0': vx0, 'id1': vx1... }
        #     (2) a list of matches [ match0, match1, ... ], where
        #         each match is [vx0, vx1, ...]

        self.ids = list(qgraph.pmap['ids'].keys())

        if len(results) > 0 and type(results[0]) == dict:
                # convert format 1 to format 2
                self.matches = []
                for  m in results:
                    match = []
                    for _id in self.ids:
                        if _id not in m:
                            #raise RuntimeError(f"cannot find id '{_id }' in result-set!")
                            match.append(None)
                        # we store the result as 'vx' instead of cnode, since the
                        # cnode could be removed, and we lose the ability to track
                        # it
                        else:
                            match.append(m[_id])
                    self.matches.append(match)
        else:
            self.matches = copy.copy(results)

        self.qgraph = qgraph
        self.g = g

        self.vs = set(g.vertices) # track
        self.__iter = None

    def apply(self, action, **kwargs):
        return action(self, **kwargs)

    def _create_match_obj(self, ids, match):
        class Result:
            def __init__(self, qrs:QueryResultSet, ids, match):

                for _id in match:
                    if _id is not None:
                        if not qrs.g.check_vx(_id):
                            raise RuntimeError("Query result corruption: vertex [%d] has been removed!" % _id)

                self.result = dict(zip(ids, match))

                for r in self.result:
                    vx = self.result[r]
                    if vx is None:
                        self.__setattr__(r, None)
                    else:
                        self.__setattr__(r, qrs.g[self.result[r]])

            @property
            def match(self):
                return self.result

            def __getitem__(self, item):
                return getattr(self, item)

            def __repr__(self):
                return str(self.result)

        return Result(self, ids, match)

    def __repr__(self):

        g_vs = set(self.g.vertices)

        def fmt(match):
            fmatch = []

            for vx in match:
                if vx not in g_vs:
                    fm = f"{Fore.LIGHTRED_EX}({vx}){Style.RESET_ALL}"
                else:
                    fm = f"{Fore.LIGHTBLACK_EX}{vx}{Style.RESET_ALL}"
                fmatch.append(fm)
            return fmatch

        data = [ fmt(match) for match in self.matches ]
        out = tabulate(data, headers=self.ids)
        return out

    @property
    def removed(self):
        current_vs = set(self.g.vertices)
        return self.vs - current_vs


    @property
    def inserted(self):
        current_vs = set(self.g.vertices)
        return current_vs - self.vs

    def __iter__(self):
        self.__iter = iter(self.matches)

        return self

    def __next__(self):
        match = next(self.__iter)
        return self._create_match_obj(self.ids, match)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, item):
        return self._create_match_obj(self.ids, self.matches[item])















