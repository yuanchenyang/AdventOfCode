# Standard Library
import doctest
import sys
import string
from dataclasses import dataclass
from math import prod
from itertools import zip_longest, pairwise, takewhile, islice, starmap, chain, cycle
from functools import cmp_to_key, cache, cached_property
from collections import deque, namedtuple, defaultdict
from operator import *
from heapq import heappush, heappop

# Other Libraries
import z3
import picos as pic
import sympy as sp

# Local Imports
from utils import *
from test_inputs import *
P = Point

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

priorities = dict(zip(string.ascii_letters, range(1, 53)))

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
    rope = [P(0, 0)] * k
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
    elev = dict(zip(string.ascii_lowercase, range(26))) | dict(S=0, E=25)
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
    g = set(P(x, y)
            for path in s.strip().split('\n')
            for (x1,y1), (x2,y2) in map(sorted, pairwise(re_lines('(\d+),(\d+)', path)))
            for x in range(x1, x2+1) for y in range(y1, y2+1))
    priority = [P(0, 1), P(-1, 1), P(1, 1)]
    bottom = max(y for x, y in g) + offset

    def fall():
        cur = P(500, 0)
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
    return day_14_common(s, 1, lambda cur, bottom: cur != P(500, 0)) + 1

day_15_regex = 'Sensor at x=([-\d]+), y=([-\d]+): closest beacon is at x=([-\d]+), y=([-\d]+)'

def day_15_common(s, delta):
    S = {P(sx, sy): P(bx, by)
         for sx, sy, bx, by in re_lines(day_15_regex, s)}
    return {si: l1_dist(si, bi) + delta(si) for si, bi in S.items()}

def day_15a(s, y=2000000):
    '''
    >>> day_15a(day_15_test_input, y=10)
    26
    '''
    intervals = [sp.Interval(si[0] - r, si[0] + r)
                 for si, r in day_15_common(s, lambda x: -abs(x[1] - y)).items()
                 if r > 0]
    return sp.Union(*intervals).measure

def day_15b(s, limit=4000000):
    '''
    >>> day_15b(day_15_test_input, limit=20)
    56000011
    '''
    sensors = day_15_common(s, lambda x: 0)
    x, y = z3.Int('x'), z3.Int('y')
    s = z3.Solver()
    s.add(x >= 0, y >= 0, x <= limit, y <= limit)
    for (sx, sy), d in sensors.items():
        s.add(z3.Or(   (x-sx) + (y-sy) > d
                    , -(x-sx) - (y-sy) > d
                    ,  (x-sx) - (y-sy) > d
                    , -(x-sx) + (y-sy) > d))
    s.check()
    return s.model().evaluate(x*4000000 + y)

day_16_regex = 'Valve (\w\w) has flow rate=(\d+); tunnels* leads* to valves* ([\w, ]+)'

def day_16_common(s):
    Node = namedtuple('Node', ('rate', 'dests'))
    nodes = {src: Node(rate, tunnels.split(', '))
             for src, rate, tunnels in re_lines(day_16_regex, s)}
    init = frozenset(label for label, node in nodes.items() if node.rate > 0)
    return nodes, init

def day_16a(s):
    '''
    >>> day_16a(day_16_test_input)
    1651
    '''
    nodes, init = day_16_common(s)
    @cache
    def solve(cur, time, off):
        if time == 0 or len(off) == 0:
            return 0
        sols = [solve(dest, time-1, off) for dest in nodes[cur].dests]
        if cur in off:
            sols.append((time - 1) * nodes[cur].rate
                        + solve(cur, time - 1, off - set([cur])))
        return max(sols)
    return solve('AA', 30, init)

def day_16b(s):
    '''
    >>> day_16b(day_16_test_input)
    1707
    '''
    T = 4
    ps = 'C'
    nodes, init = day_16_common(s)

    vs = {(p, t, n): z3.Int(f'{p}_{t}_{n}')
          for p in ps for t in range(T+1) for n in nodes}
    on = {(n, t): z3.Int(f'{n}_{t}') for n in init for t in range(T)}
    opt = z3.Optimize()
    for (n, t), v in on.items():
        opt.add(v >= 0, v <= 1)
        opt.add(v == sum(vs[(p, t, n)]*vs[(p, t+1, n)] for p in ps))
    for n in init:
        opt.add(sum(on[(n, t)] for t in range(T)) <= 1)
    for (p, t, n), var in vs.items():
        opt.add(var >= 0, var <= 1)
        if t == 0: continue
        dests = sum(vs[(p, t-1, d)] for d in nodes[n].dests)
        if n in init: # Maybe turn on
            dests += on[(n, t-1)]
        if n == 'AA' and t == 3:
            #breakpoint()
            pass
        opt.add(dests == var)
    for n in nodes:
        for p in ps:
            opt.add(vs[(p, T, n)] == (n == 'AA'))
    cost = sum(z3.If(v == 1, nodes[n].rate*t, 0) for (n, t), v in on.items())
    h = opt.maximize(cost)
    print(opt)
    print(opt.check())
    m = opt.model()
    # for i, v in on.items():
    #     print(i, m.evaluate(v))
    # for i, v in vs.items():
    #     print(i, m.evaluate(v))
    return opt.lower(h)

def day_17_common(s, n, n_history=100):
    blocks = cycle(enumerate([(P(0, 0), P(1, 0), P(2, 0), P(3, 0))
                             ,(P(1, 0), P(0, 1), P(1, 1), P(1, 2), P(2, 1))
                             ,(P(0, 0), P(1, 0), P(2, 0), P(2, 1), P(2, 2))
                             ,(P(0, 0), P(0, 1), P(0, 2), P(0, 3))
                             ,(P(0, 0), P(1, 0), P(0, 1), P(1, 1))]))
    dx = {'>': P(1, 0), '<': P(-1, 0)}

    def drop():
        grid = set()
        update = lambda p, cur: tuple(c + p for c in cur)
        valid = lambda block: all(p not in grid and p[1] >= 0 and 0 <= p[0] <= 6
                                  for p in block)
        cur = update(P(2, 3), next(blocks)[1])
        bottom = -1
        for i, c in cycle(enumerate(s.strip())):
            if valid(shift := update(dx[c], cur)):
                cur = shift
            if valid(down := update(P(0, -1), cur)):
                cur = down
            else:
                grid.update(cur)
                db = max(py for px, py in grid) - bottom
                j, block = next(blocks)
                yield i, j, db
                bottom += db
                cur = update(P(2, bottom + 4), block)

    ## Following suffices for part a
    # return sum(db for _, _, db in islice(drop(), n))

    history, states = [], {}
    for i, j, db in drop():
        history.append(db)
        if (state := (i, j, tuple(history[-n_history:]))) in states:
            break
        states[state] = None # Using dict as ordered set

    prefix = list(states).index(state)
    l = len(states) - prefix
    d, r = (n-prefix) // l, (n-prefix) % l
    head, repeat = history[:prefix], history[prefix:-1]
    return sum(head) + d * sum(repeat) + sum(repeat[:r])

def day_17a(s):
    '''
    >>> day_17a(day_17_test_input)
    3068
    '''
    return day_17_common(s, 2022)

def day_17b(s):
    '''
    >>> day_17b(day_17_test_input)
    1514285714288
    '''
    return day_17_common(s, 1000000000000)

def day_18_common(s):
    dirs = [P(-1,0,0), P(1,0,0), P(0,-1,0), P(0,1,0), P(0,0,-1), P(0,0,1)]
    points = set(P(*line) for line in Array(s, ','))
    return dirs, points

def day_18a(s):
    '''
    >>> day_18a(day_18_test_input)
    64
    '''
    dirs, points = day_18_common(s)
    return sum(1 for p in points for d in dirs if p + d not in points)

def day_18b(s):
    '''
    >>> day_18b(day_18_test_input)
    58
    '''
    dirs, points = day_18_common(s)
    def dfs(start, limit):
        visited = set()
        to_visit = [(start, 0)]
        while len(to_visit) > 0:
            cur, dist = to_visit.pop()
            if dist > limit:
                return False
            visited.add(cur)
            for d in dirs:
                n = cur + d
                if n not in visited and n not in points:
                    to_visit.append((n, dist+1))
        return visited

    adjacent = set(p + d for p in points for d in dirs)
    interior = set(points)
    for p in adjacent:
        if p not in interior and (visited := dfs(p, len(points))):
            interior.update(visited)
    return sum(1 for p in points for d in dirs if p + d not in interior)

day_19_regex = 'Blueprint (\d+): Each ore robot costs (\d+) ore. '\
'Each clay robot costs (\d+) ore. Each obsidian robot costs (\d+) ore and (\d+) clay. '\
'Each geode robot costs (\d+) ore and (\d+) obsidian.'

State = namedtuple(
    'State',
    ['t',                            # timestep
     'n_or', 'n_cl', 'n_ob', 'n_ge', # no. of each resource at end of timestep t
     'r_or', 'r_cl', 'r_ob', 'r_ge'  # no. of robots at end of timestep t
     ],
    defaults = [0]*8
)

def day_19_common(lst, time):
    update = lambda s: s._replace(t = s.t - 1,
                                  n_or = s.n_or + s.r_or, n_cl = s.n_cl + s.r_cl,
                                  n_ob = s.n_ob + s.r_ob, n_ge = s.n_ge + s.r_ge)
    def collect(s):
        while s.t > 0:
            yield s
            s = update(s)
    def upper_bound(s):
        return s.n_ge + (s.t)*(s.t-1)//2 + s.t*s.r_ge
    for _, or_or, cl_or, ob_or, ob_cl, ge_or, ge_ob in lst:
        max_or = max(or_or, cl_or, ob_or, ge_or)
        states = [State(t=time-1, n_or=1, r_or=1)]
        best = 0
        while len(states) > 0:
            cur = states.pop()
            if cur.t == 0:
                best = max(best, cur.n_ge)
                continue
            if upper_bound(cur) <= best:
                continue
            times = {}
            for S in collect(cur):
                if 'ge' not in times and S.n_or >= ge_or and S.n_ob >= ge_ob:
                    times['ge'] = update(S._replace(
                        n_or = S.n_or - ge_or,
                        n_ob = S.n_ob - ge_ob
                    ))._replace(r_ge = S.r_ge + 1)
                if 'ob' not in times and S.n_or >= ob_or and S.n_cl >= ob_cl\
                   and S.r_ob < ge_ob:
                    times['ob'] = update(S._replace(
                        n_or = S.n_or - ob_or,
                        n_cl = S.n_cl - ob_cl
                    ))._replace(r_ob = S.r_ob + 1)
                if 'cl' not in times and S.n_or >= cl_or and S.r_cl < ob_cl:
                    times['cl'] = update(S._replace(n_or = S.n_or - cl_or))\
                        ._replace(r_cl = S.r_cl + 1)
                if 'or' not in times and S.n_or >= or_or and S.r_or < max_or:
                    times['or'] = update(S._replace(n_or = S.n_or - or_or))\
                        ._replace(r_or = S.r_or + 1)
            times['end'] = update(S)
            for key in ('ob', 'cl', 'or', 'ge', 'end'):
                if key in times:
                    states.append(times[key])
        yield best

def day_19a(s):
    '''
    >>> day_19a(day_19_test_input)
    33
    '''
    res = day_19_common(re_lines(day_19_regex, s), 24)
    return sum(best*i for best, i in enumerate(res, 1))

def day_19b(s):
    '''
    >>> day_19b(day_19_test_input)
    3472
    '''
    return prod(day_19_common(re_lines(day_19_regex, s)[:3], 32))

def day_20_common(s, mul, times):
    lst = [(c*mul, i) for i, c in enumerate(List(s))]
    order = lst[:]
    for _ in range(times):
        for c in order:
            i = lst.index(c)
            lst = lst[i+1:] + lst[:i]
            j = c[0] % len(lst)
            lst = lst[:j] + [c] + lst[j:]
    return sum(lst[i][0] for d in (1000, 2000, 3000) for i in range(len(lst))
               if lst[(i-d) % len(lst)][0] == 0)

def day_20a(s):
    '''
    >>> day_20a(day_20_test_input)
    3
    '''
    return day_20_common(s, 1, 1)

def day_20b(s):
    '''
    >>> day_20b(day_20_test_input)
    1623178306
    '''
    return day_20_common(s, 811589153, 10)

def day_21_common(s):
    env = dict(Array(s, ': '))
    def expand(expr, env):
        if expr in env:
            return expand(env[expr], env)
        if type(expr) == str:
            a, op, b = expr.split()
            return f'(({expand(a, env)}) {op} ({expand(b, env)}))'
        return f'sp.sympify({expr})'
    return expand, env

def day_21a(s):
    '''
    >>> day_21a(day_21_test_input)
    152
    '''
    expand, env = day_21_common(s)
    return eval(expand('root', env))

def day_21b(s):
    '''
    >>> day_21b(day_21_test_input)
    301
    '''
    expand, env = day_21_common(s)
    a, _, b = env['root'].split()
    env['root'] = f'{a} - {b}'
    x = sp.symbols('x')
    env['humn'] = x
    return sp.solve(eval(expand('root', env)), x)[0]

def day_22_hardcode():
    cube = [
      ([P(50,50+i) for i in range(50)], [P(i,100) for i in range(50)], 2, 3)
     ,([P(50,49-i) for i in range(50)], [P(0,100+i) for i in range(50)], 2, 2)
     ,([P(50+i,0) for i in range(50)], [P(0,150+i) for i in range(50)], 3, 2)
     ,([P(100+i,0) for i in range(50)], [P(i,199) for i in range(50)], 3, 1)
     ,([P(100+i,49) for i in range(50)], [P(99,50+i) for i in range(50)], 1, 0)
     ,([P(149,49-i) for i in range(50)], [P(99,100+i) for i in range(50)], 0, 0)
     ,([P(49,150+i) for i in range(50)], [P(50+i,149) for i in range(50)], 0, 1)
    ]
    glue = {}
    other = lambda d: (d+2) % 4
    for l0, l1, d0, d1 in cube:
        for p0, p1 in zip(l0, l1):
            glue[(p0, d0)] = (p1, other(d1))
            glue[(p1, d1)] = (p0, other(d0))
    return glue

def day_22_common(s, hardcode=False):
    dirs = (P(1, 0), P(0, 1), P(-1, 0), P(0,-1))
    turn = dict(R=1, L=-1)

    l0, l1 = s.split('\n\n')
    l0 = l0.split('\n')
    l_max = max(map(len, l0))
    grid = Grid('\n'.join(l.ljust(l_max) for l in l0))
    cur, f = P(grid[0].index('.'), 0), 0

    glue = day_22_hardcode()
    def cube_hardcode(p, f):
        return glue.get((p, f), (p + dirs[f], f))
    def torus(p, f):
        while grid.get(nxt := ((p + dirs[f]) % grid.dim)) == ' ':
            p = nxt
        return nxt, f
    update = cube_hardcode if hardcode else torus

    for i in chain(*re_lines('([RL]|\d+)', l1)):
        if type(i) == int:
            for _ in range(i):
                nxt, nf = update(cur, f)
                if grid.get(nxt) == '#': break
                cur, f = nxt, nf
        else:
            f = (f + turn[i]) % len(dirs)
    return 1000 * (cur[1]+1) + 4 * (cur[0]+1) + f

def day_22a(s):
    '''
    >>> day_22a(day_22_test_input)
    6032
    '''
    return day_22_common(s)

def day_22b(s):
    '''
    >>> day_22b(open('inputs/22.in').read())
    11451
    '''
    return day_22_common(s, hardcode=True)

def day_23_common(s, rounds=10, find_fixed=False):
    S, N, E, W, SE, NE, SW, NW = all_dirs
    valid = [(N, (N,NE,NW)), (S, (S,SE,SW)), (W, (W,NW,SW)), (E, (E,NE,SE))]

    elves = set(P(x, y) for y, line in enumerate(List(s))
                        for x, c in enumerate(line)
                        if c == '#')
    def no(elf, dirs):
        return all((elf + d) not in elves for d in dirs)
    for i in range(rounds):
        proposed = defaultdict(list)
        for elf in elves:
            if no(elf, all_dirs):
                proposed[elf].append(elf)
                continue
            for j in range(4):
                d, dirs = valid[(i+j) % 4]
                if no(elf, dirs):
                    proposed[elf + d].append(elf)
                    break
            else:
                proposed[elf].append(elf)
        new = set()
        for dest, sources in proposed.items():
            if len(sources) == 1:
                new.add(dest)
            else:
                new.update(sources)
        if find_fixed and len(elves - new) == 0:
            return i + 1
        elves = new
    Xs, Ys = zip(*elves)
    return (max(Xs) - min(Xs) + 1) * (max(Ys) - min(Ys) + 1) - len(elves)

def day_23a(s):
    '''
    >>> day_23a(day_23_test_input)
    110
    '''
    return day_23_common(s)

def day_23b(s):
    '''
    >>> day_23b(day_23_test_input)
    20
    '''
    return day_23_common(s, rounds=50000, find_fixed=True)

def day_24_common(s):
    dirs = {'>': P(1, 0), '<': P(-1, 0), '^': P(0, -1), 'v': P(0, 1)}
    bliz = defaultdict(list)
    for y, line in enumerate(List(s)[1:-1]):
        for x, c in enumerate(line[1:-1]):
            if c != '.':
                bliz[P(x, y)].append(dirs[c])
    xmax, ymax = x+1, y+1
    start, end = P(0, -1), P(x, ymax)
    def spaces(bliz):
        while True:
            yield {P(x, y) for x in range(xmax) for y in range(ymax)
                   if P(x, y) not in bliz} | set([start, end])
            new_bliz = defaultdict(list)
            for b, ds in bliz.items():
                for d in ds:
                    new_bliz[(b+d) % P(xmax, ymax)].append(d)
            bliz = new_bliz
    return LazyList(spaces(bliz)), start, end

def day_24_search(spaces, start, end, start_time):
    heuristic = lambda p: l1_dist(p, end)
    to_visit = [(heuristic(start), start, start_time)]
    visited = set()
    while True:
        h, cur, t = heappop(to_visit)
        visited.add((h, cur, t))
        if cur == end:
            return t
        for d in cardinal_dirs + [P(0,0)]:
            if (new := cur + d) in spaces[t + 1]:
                if (item := (t + 1 + heuristic(new), new, t + 1)) not in visited:
                    heappush(to_visit, item)
                    visited.add(item)

def day_24a(s):
    '''
    >>> day_24a(day_24_test_input)
    18
    '''
    spaces, start, end = day_24_common(s)
    return day_24_search(spaces, start, end, 0)

def day_24b(s):
    '''
    >>> day_24b(day_24_test_input)
    54
    '''
    spaces, start, end = day_24_common(s)
    t1 = day_24_search(spaces, start, end, 0)
    t2 = day_24_search(spaces, end, start, t1)
    return day_24_search(spaces, start, end, t2)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test':
            fn_to_test = globals()['day_' + sys.argv[2]]
            doctest.run_docstring_examples(fn_to_test, globals())
        else:
            print(globals()['day_' + sys.argv[1]](sys.stdin.read()))
    else:
        doctest.testmod()
