from typing import Iterable


hr = "--------------------------------------------------"
A4 = '            '
A3 = '         '
A2 = '       '
A1 = '     '
A0 = '   '


class Debug:
    def __init__(self, verbose, logger=None):
        self.verbose = verbose
        self.logger = logger

    def dbg(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)

    def hr(self):
        self.dbg(hr)

    def bm(self, msg):
        row = len(msg) + 2
        h = ''.join(['+'] + ['-' * row] + ['+'])
        result = h + '\n'"| " + msg + " |"'\n' + h
        self.dbg(result)


class DebugFCI(Debug):
    def __init__(self, verbose, logger=None):
        super().__init__(verbose, logger)
        self.verbose = verbose
        self.logger = logger

    @staticmethod
    def _len(iterable: Iterable):
        return len(list(filter(None, iterable)))

    def empty_set(self, i, dseps, x, y):
        dbg_msg = "{}+ [{}/{}] d-sep. sets for {}⫫{} is EMPTY"
        self.dbg(dbg_msg.format(A2, i + 1, self._len(dseps[x]) + 1, x, y))

    def d_seps(self, i, dseps, dsep_combinations, x, y):
        dbg_msg = "{}+ [{}/{}] d-sep. sets for {}⫫{} has {} combs: "
        self.dbg(dbg_msg.format(A2, i, self._len(dseps[x]), x, y,
                                self._len(dsep_combinations)), end="")
        # dbg_msg = "{}> {}⫫{} | {} combs: "
        # oo(dbg_msg.format(A3, x, y, len(dsep_combinations)), end="")
        self.dbg(','.join(
            [f"({','.join(c)})" for c in dsep_combinations if bool(c)]),
            sep="")

    def y(self, x, y, ny, dseps, neighbors):
        self.dbg(f"{A0}+ y = {y}; {ny + 1}/{len(neighbors)} ")
        dbg_msg = "{}> Check {}⫫{} | over {} d-separation sets: "
        self.dbg(dbg_msg.format(A1, x, y, self._len(dseps[x])), end="")
        self.dbg(f"({','.join(dseps[x])})")

    def neighbors(self, x, lx, neighbors, pag):
        nodes_left = sorted(list(set(pag) - {x}))
        if not len(neighbors):
            self.dbg(f"{A0}> NO neighbors")
        else:
            self.dbg(f" + x = {x}; {lx}/{len(nodes_left)}")
            ns = ",".join([i for i in neighbors])
            self.dbg(f"{A0}> Explore {len(neighbors)} neighb. of {x}: ({ns})")

    def interrupt(self):
        self.dbg("--- ⚡️ ---  BREAKING: Independence found")

    @staticmethod
    def remove_redundancies(orig: str, dest: str, stack) -> bool:
        if orig < dest:
            return False
        matches_dest = list(map(lambda t: t[0] == dest, stack[orig]))
        if any(matches_dest):
            for i, is_matching in enumerate(matches_dest):
                if is_matching:
                    del stack[orig][i]
            return True

    def stack(self, stack):
        if not len(stack):
            return
        self.dbg(f"\nACTIONS STACK ({len(stack.items())})\n{hr}")
        for from_node, tuples in sorted(stack.items()):
            line_printed = False
            first_element = True
            for to_node, cs in tuples:
                if first_element:
                    self.dbg(f"{from_node}: ", end="")
                    line_printed = True
                    first_element = False
                self.remove_redundancies(to_node, from_node, stack)
                if cs:
                    self.dbg(f"{to_node}|({','.join(cs)})", end="")
                else:
                    self.dbg(f"{to_node}", end="")
                self.dbg(",", end="")
            if line_printed:
                self.dbg("")
        self.hr()
