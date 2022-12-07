import doctest
import sys
from itertools import zip_longest

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
    return sum(sorted(day_1_input(s))[-3:])

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
    return day_5_common(s, preprocess=lambda x:x)

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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test':
            fn_to_test = globals()['day_' + sys.argv[2]]
            doctest.run_docstring_examples(fn_to_test, globals())
        else:
            print(globals()['day_' + sys.argv[1]](sys.stdin.read()))
    else:
        doctest.testmod()
