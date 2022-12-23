import re
from functools import reduce
from operator import mul

letters = 'abcdefghijklmnopqrstuvwxyz'
upper_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class Grid:
    def __init__(self, s):
        self.grid = [[Token(c) for c in row] for row in s.splitlines()]
        self.xlen = len(self.grid[0])
        self.ylen = len(self.grid)
        self.dim = Point(self.xlen, self.ylen)

    def __getitem__(self, i):
        return self.grid[i]

    def get(self, p):
        x, y = p
        return self.grid[y][x]

    def iter_points(self):
        for x in range(self.xlen):
            for y in range(self.ylen):
                yield Point(x, y), self.grid[y][x]

    def __contains__(self, p):
        x, y = p
        return x in range(self.xlen) and y in range(self.ylen)

def transpose(grid):
    return list(zip(*grid))

def flip(grid):
    return [list(reversed(row)) for row in grid]

def transflip(grid):
    return flip(transpose(grid))

class Point(tuple):
    def __new__(self, *args):
        return super().__new__(self, args)
    def __add__(self, other):
        return Point(*[s + o for s, o in zip(self,other)])
    def __neg__(self):
        return Point(*[-s for s in self])
    def __sub__(self, other):
        return self + (-other)
    def __mod__(self, other):
        return Point(*[s % o for s, o in zip(self,other)])
    def __repr__(self):
        return 'Point' + super().__repr__()

# S, N, E, W
cardinal_dirs = [Point(0, 1), Point(0, -1), Point(1,  0), Point(-1,  0)]
# SE, NE, SW, NW
ordinal_dirs =  [Point(1, 1), Point(1, -1), Point(-1, 1), Point(-1, -1)]
# S, N, E, W, SE, NE, SW, NW
all_dirs = cardinal_dirs + ordinal_dirs

def l1_dist(x, y):
    return sum(abs(i) for i in (x-y))

def linf_dist(x, y):
    return max(abs(i) for i in (x-y))

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
    return [Token(t) for t in str(s).strip().split(sep)]

def List(s):
    return [Token(t.strip()) for t in s.splitlines()]

def ListOfList(s, sep='\n\n'):
    return [List(l) for l in s.split(sep)]

def re_lines(regex, s):
    return [[Token(t) for t in match.groups()] for match in re.finditer(regex, s)]

def grouped(seq, n):
    '''iterator -> iterator of iterators with groups of size n'''
    return zip(*[iter(seq)]*n)

def llen(seq):
    return len(list(seq))

def topk(seq, k):
    return sorted(seq)[-k:]

def first(pair):
    return pair[0]

def second(pair):
    return pair[1]

iden = lambda x: x
