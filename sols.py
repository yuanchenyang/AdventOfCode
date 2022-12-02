import doctest
import sys

from utils import *

def day_1_input(s):
    return [sum(List(lst)) for lst in s.split('\n\n')]

def day_1a(s):
    return max(day_1_input(s))

def day_1b(s):
    return sum(sorted(day_1_input(s))[-3:])

day_2_test_input='''A Y
B X
C Z
'''

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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-test':
            fn_to_test = globals()['day_' + sys.argv[2]]
            doctest.run_docstring_examples(fn_to_test, globals())
        else:
            print(globals()['day_' + sys.argv[1]](sys.stdin.read()))
    else:
        doctest.testmod()
