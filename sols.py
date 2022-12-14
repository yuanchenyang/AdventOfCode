import doctest
import sys

from dataclasses import dataclass
from math import prod
from itertools import zip_longest, pairwise, takewhile, islice, starmap, chain
from functools import cmp_to_key
from collections import deque
from operator import *

from utils import *
from test_inputs import *

def day_1_input(s):
    return [sum(lst) for lst in ListOfList(s)]

def day_1a(s):
    '''
    >>> day_1a(day_1_test_input)
    24000
    '''
    return max(day_1_input(s))

def day_1b(s):
    '''
    >>> day_1b(day_1_test_input)
    45000
    '''
    return sum(topk(day_1_input(s), 3))

def day_2a(s):
    '''
    >>> day_2a(day_2_test_input)
    15
    '''
    p1 = dict(A=0, B=1, C=2)
    p2 = dict(X=0, Y=1, Z=2)
    # outcome[p1][p2] = score of result
    outcome = [[3, 6, 0],
               [0, 3, 6],
               [6, 0, 3]]
    return sum(outcome[p1[i]][p2[j]] + p2[j] + 1 for i, j in Array(s))

def day_2b(s):
    '''
    >>> day_2b(day_2_test_input)
    12
    '''
    p1 = dict(A=0, B=1, C=2)
    outcome = dict(X=0, Y=1, Z=2)
    # p2[p1][outcome] = what to play
    p2 = [[2, 0, 1],
          [0, 1, 2],
          [1, 2, 0]]
    return sum(p2[p1[i]][outcome[j]] + 1 + outcome[j]*3 for i, j in Array(s))

priorities = dict(zip(letters + upper_letters, range(1, 53)))

def day_3a(s):
    '''
    >>> day_3a(day_3_test_input)
    157
    '''
    def intersect(sack):
        middle = len(sack) // 2
        return (set(sack[:middle]) & set(sack[middle:])).pop()
    return sum(priorities[intersect(l)] for l in List(s))

def day_3b(s):
    '''
    >>> day_3b(day_3_test_input)
    70
    '''
    return sum(priorities[(set(a) & set(b) & set(c)).pop()]
               for a, b, c in grouped(List(s), 3))

day_4_regex = '(\d+)-(\d+),(\d+)-(\d+)'

def contains(a1, a2, b1, b2):
    return a1 <= b1 <= a2 and a1 <= b2 <= a2

def fully_contains(a1, a2, b1, b2):
    return contains(a1, a2, b1, b2) or contains(b1, b2, a1, a2)

def day_4a(s):
    '''
    >>> day_4a(day_4_test_input)
    2
    '''
    return sum(fully_contains(*inputs) for inputs in re_lines(day_4_regex, s))

def overlaps(a1, a2, b1, b2):
    return not (a2 < b1 or b2 < a1)

def day_4b(s):
    '''
    >>> day_4b(day_4_test_input)
    4
    '''
    return sum(overlaps(*inputs) for inputs in re_lines(day_4_regex, s))

day_5_regex = 'move (\d+) from (\d) to (\d)'

def day_5_common(s, preprocess):
    stacks, instructions = s.split('\n\n')
    rows = reversed(stacks.splitlines())
    crates = {int(line[0]): list(filter(str.isupper, line))
              for line in zip_longest(*rows, fillvalue=' ')
              if line[0].isdigit()}

    for num, src, dest in re_lines(day_5_regex, instructions):
        crates[dest].extend(preprocess([crates[src].pop() for _ in range(num)]))
    return ''.join(crates[i].pop() for i in sorted(crates))

def day_5a(s):
    '''
    >>> day_5a(day_5_test_input)
    'CMZ'
    '''
    return day_5_common(s, preprocess=iden)

def day_5b(s):
    '''
    >>> day_5b(day_5_test_input)
    'MCD'
    '''
    return day_5_common(s, preprocess=reversed)

def day_6_common(s, n):
    for i, seq in enumerate(zip(*[s[i:] for i in range(n)])):
        if len(set(seq)) == n:
            return i + n

def day_6a(s):
    '''
    >>> [day_6a(s) for s in day_6_test_input.splitlines()]
    [7, 5, 6, 10, 11]
    '''
    return day_6_common(s, 4)

def day_6b(s):
    '''
    >>> [day_6b(s) for s in day_6_test_input.splitlines()]
    [19, 23, 23, 29, 26]
    '''
    return day_6_common(s, 14)

class Dir:
    def __init__(self, parent=None):
        self.parent = parent
        self.size = 0
        self.contents = {}

    def add_file(self, name, size):
        cur = self
        while cur != None:
            cur.size += size
            cur = cur.parent

def day_7_parse(s):
    cur = Dir()
    dirs = [cur]
    for command in s.strip().split('\n$ ')[1:]:
        match Array(command):
            case [['ls'], *rest]:
                for info, name in rest:
                    if info == 'dir':
                        cur.contents[name] = Dir(cur)
                        dirs.append(cur.contents[name])
                    else:
                        cur.add_file(name, info)
            case [['cd', '..']]:
                cur = cur.parent
            case [['cd', d]]:
                cur = cur.contents[d]
    return dirs

def day_7a(s):
    '''
    >>> day_7a(day_7_test_input)
    95437
    '''
    return sum(d.size for d in day_7_parse(s) if d.size < 100000)

def day_7b(s):
    '''
    >>> day_7b(day_7_test_input)
    24933642
    '''
    sizes = [d.size for d in day_7_parse(s)]
    unused = 70000000 - sizes[0]
    return min(filter(lambda x: x + unused >= 30000000, sizes))

def day_8a_n_squared(s):
    ''' O(N^2) implementation, but doesn't generalize to 8b
    >>> day_8a_n_squared(day_8_test_input)
    21
    '''
    def visrow(row):
        tallest = -1
        for i, n in enumerate(row):
            if n > tallest:
                yield i
                tallest = n
    S = Grid(s).grid
    n = len(S)-1
    transforms = [(iden, (lambda i, j: (i, j))),
                  (transpose, (lambda i, j: (j, i))),
                  (flip, (lambda i, j: (i, n-j))),
                  (transflip, (lambda i, j: (n-j, i)))]
    return len(set(g(i, j) for f, g in transforms
                   for i, row in enumerate(f(S)) for j in visrow(row)))

def day_8_common(s):
    grid = Grid(s)
    def iterdir(x, y, dx, dy):
        while (x := x+dx, y := y+dy) in grid:
            yield grid[y][x]
    return grid, iterdir

def day_8a(s):
    '''
    >>> day_8a(day_8_test_input)
    21
    '''
    grid, iterdir = day_8_common(s)
    return len(set((x, y) for x in range(grid.xlen) for y in range(grid.ylen)
                          for dx, dy in cardinal_dirs
                          if all(grid[y][x] > e for e in iterdir(x, y, dx, dy))))

def day_8b(s):
    '''
    >>> day_8b(day_8_test_input)
    8
    '''
    grid, iterdir = day_8_common(s)
    def see(e, lst):
        for i in lst:
            yield i
            if i >= e: break
    return max(prod(llen(see(grid[y][x], iterdir(x, y, dx, dy)))
                    for dx, dy in cardinal_dirs)
               for x in range(grid.xlen) for y in range(grid.ylen))

def day_9_common(s, k):
    rope = [Point(0, 0)] * k
    deltas = dict(zip('RLUD', cardinal_dirs))
    visited = set([rope[-1]])

    def follow(H, T):
        if linf_dist(T, H) > 1:
            T += min(all_dirs, key=lambda x: l1_dist(T + x, H))
        return T

    for d, n in Array(s):
        for _ in range(n):
            rope[0] += deltas[d]
            for i in range(1,k):
                rope[i] = follow(rope[i-1], rope[i])
            visited.add(rope[-1])
    return len(visited)

def day_9a(s):
    '''
    >>> day_9a(day_9_test_input)
    13
    '''
    return day_9_common(s, 2)

def day_9b(s):
    '''
    >>> day_9b(day_9_test_input)
    1
    >>> day_9b(day_9_test_input2)
    36
    '''
    return day_9_common(s, 10)

def day_10_common(s):
    X = 1
    for row in Array(s):
        yield X
        if row[0] == 'addx':
            yield X
            X += row[1]

def day_10a(s):
    '''
    >>> day_10a(day_10_test_input)
    13140
    '''
    return sum((i+1) * n for i, n in islice(enumerate(day_10_common(s)), 19, None, 40))

def day_10b(s):
    '''
    >>> day_10b(day_10_test_input)
    ##..##..##..##..##..##..##..##..##..##..
    ###...###...###...###...###...###...###.
    ####....####....####....####....####....
    #####.....#####.....#####.....#####.....
    ######......######......######......####
    #######.......#######.......#######.....
    '''
    seq = grouped(('#' if abs(i%40 - n) < 2 else '.'
                   for i, n in enumerate(day_10_common(s))),
                  40)
    print('\n'.join(''.join(line) for line in seq))

day_11_re = '''Monkey (\d+):
  Starting items: (.*)
  Operation: new = (.*)
  Test: divisible by (\d+)
    If true: throw to monkey (\d+)
    If false: throw to monkey (\d+)'''

@dataclass
class Monkey:
    items: list[int]
    op: str
    test: int
    dest: tuple[int]
    inspected: int=0

def day_11_common(s, n, div):
    monkeys = [Monkey(Line(l, ', '), op, test, (t, f))
               for _, l, op, test, t, f in re_lines(day_11_re, s)]
    lcm = prod(m.test for m in monkeys)
    for _ in range(n):
        for m in monkeys:
            for old in m.items:
                m.inspected += 1
                new = (eval(m.op, globals(), locals()) // div) % lcm
                monkeys[m.dest[new % m.test != 0]].items.append(new)
            m.items = []
    return prod(topk([m.inspected for m in monkeys], 2))

def day_11a(s):
    '''
    >>> day_11a(day_11_test_input)
    10605
    '''
    return day_11_common(s, 20, 3)

def day_11b(s):
    '''
    >>> day_11b(day_11_test_input)
    2713310158
    '''
    return day_11_common(s, 10000, 1)

def day_12_common(s, frontier):
    g = Grid(s)
    elev = dict(zip(letters, range(26))) | dict(S=0, E=25)
    to_visit = deque((p, 0) for p, elem in g.iter_points() if elem in frontier)
    visited = set(map(first, to_visit))

    while len(to_visit) != 0:
        cur, n = to_visit.popleft()
        if g.get(cur) == 'E':
            return n
        for new in [cur + d for d in cardinal_dirs]:
            if new in g and new not in visited and \
               elev[g.get(new)] <= elev[g.get(cur)] + 1:
                visited.add(new)
                to_visit.append((new, n+1))

def day_12a(s):
    '''
    >>> day_12a(day_12_test_input)
    31
    '''
    return day_12_common(s, 'S')

def day_12b(s):
    '''
    >>> day_12b(day_12_test_input)
    29
    '''
    return day_12_common(s, 'Sa')

def compare(l1, l2):
    match (l1, l2):
        case (int() , int() ): return l1 - l2
        case (int() , list()): return compare([l1], l2)
        case (list(), int() ): return compare(l1, [l2])
        case (list(), list()):
            for c in map(compare, l1, l2):
                if c != 0: return c
            return len(l1) - len(l2)

def day_13a(s):
    '''
    >>> day_13a(day_13_test_input)
    13
    '''
    res = [compare(eval(l1), eval(l2)) for l1, l2 in ListOfList(s)]
    return sum(i for i, r in enumerate(res, 1) if r < 0)

def day_13b(s):
    '''
    >>> day_13b(day_13_test_input)
    140
    '''
    anchors = [[[2]], [[6]]]
    res = sorted([eval(l) for l in chain(*ListOfList(s))] + anchors,
                 key=cmp_to_key(compare))
    return prod(i for i, l in enumerate(res, 1) if l in anchors)

def day_14_common(s, offset, stop):
    g = set(Point(x, y)
            for path in s.strip().split('\n')
            for (x1,y1), (x2,y2) in map(sorted, pairwise(re_lines('(\d+),(\d+)', path)))
            for x in range(x1, x2+1) for y in range(y1, y2+1))
    priority = [Point(0, 1), Point(-1, 1), Point(1, 1)]
    bottom = max(y for x, y in g) + offset

    def fall():
        cur = Point(500, 0)
        for _ in range(bottom):
            move = [cur + dx for dx in priority if cur + dx not in g]
            if len(move) == 0: break
            cur = move[0]
        return cur if stop(cur, bottom) else None

    start = len(g)
    while (rest := fall()) is not None:
        g.add(rest)
    return len(g) - start

def day_14a(s):
    '''
    >>> day_14a(day_14_test_input)
    24
    '''
    return day_14_common(s, 0, lambda cur, bottom: cur[1] < bottom)

def day_14b(s):
    '''
    >>> day_14b(day_14_test_input)
    93
    '''
    return day_14_common(s, 1, lambda cur, bottom: cur != Point(500, 0)) + 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test':
            fn_to_test = globals()['day_' + sys.argv[2]]
            doctest.run_docstring_examples(fn_to_test, globals())
        else:
            print(globals()['day_' + sys.argv[1]](sys.stdin.read()))
    else:
        doctest.testmod()
