class RSet:
    @staticmethod
    def distinct(rs, target):
        if type(target) == str:
            target = [target]

        pos = []
        for i in target:
            try:
                pos.append(rs.ids.index(i))
            except ValueError:
                raise ValueError(f"Identifier '{i}' does not exist in query results!") from None

        fmatches = []
        stored_matches = set() # keep all vx from column idt
        for match in rs.matches:
            vx_match = tuple([match[p] for p in pos])
            if vx_match not in stored_matches:
                fmatches.append(match)
                stored_matches.add(vx_match)

        qrs = rs.__class__(g=rs.g, qgraph=rs.qgraph, results=fmatches)

        return qrs

    @staticmethod
    def disjoint(rs, target):
        nodes = []
        fmatches = []

        try:
            pos = rs.ids.index(target)
        except ValueError:
            raise ValueError(f"Identifier '{target}' does not exist in query results!") from None

        for match in rs.matches:
            include_match = True
            tnode = rs.g[match[pos]]
            for node in nodes:
                if tnode.overlaps(node):
                    include_match = False
                    break
            if include_match:
                fmatches.append(match)
                nodes.append(tnode)

        qrs = rs.__class__(g=rs.g, qgraph=rs.qgraph, results=fmatches)

        return qrs



