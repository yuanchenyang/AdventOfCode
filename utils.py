import re
from functools import reduce
from operator import mul

letters = 'abcdefghijklmnopqrstuvwxyz'
upper_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
cardinal_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

class Grid:
    def __init__(self, s):
        self.grid = [[Token(c) for c in row] for row in s.splitlines()]
        self.xlen = len(self.grid[0])
        self.ylen = len(self.grid)

    def __getitem__(self, i):
        return self.grid[i]

    def __contains__(self, p):
        x, y = p
        return x in range(self.xlen) and y in range(self.ylen)

def Token(t):
    try:
        return int(t)
    except (ValueError, TypeError):
        try:
            return float(t)
        except (ValueError, TypeError):
            return t

def Array(s, sep=' '):
    return [Line(l, sep=sep) for l in s.splitlines()]

def Line(s, sep=' '):
    return [Token(t) for t in s.strip().split(sep)]

def List(s):
    return [Token(t.strip()) for t in s.splitlines()]

def ListOfList(s, sep='\n\n'):
    return [List(l) for l in s.split(sep)]

def re_tokens(regex, s):
    return [Token(t) for t in re.match(regex, s).groups()]

def re_lines(regex, s):
    return [re_tokens(regex, line) for line in s.splitlines()]

def grouped(seq, n):
    '''iterator -> iterator of iterators with groups of size n'''
    return zip(*[iter(seq)]*n)

def llen(seq):
    return len(list(seq))

def prod(seq):
    return reduce(mul, seq)
