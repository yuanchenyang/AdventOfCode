import re

letters = 'abcdefghijklmnopqrstuvwxyz'
upper_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def Token(t):
    try:
        return int(t)
    except (ValueError, TypeError):
        try:
            return float(t)
        except (ValueError, TypeError):
            return t

def Array(s, split=' '):
    return [Line(l, split=split) for l in s.splitlines()]

def Line(s, split=' '):
    return [Token(t) for t in s.strip().split(split)]

def List(s):
    return [Token(t.strip()) for t in s.splitlines()]

def re_tokens(regex, s):
    return [Token(t) for t in re.match(regex, s).groups()]

def re_lines(regex, s):
    return [re_tokens(regex, line) for line in s.splitlines()]

def grouped(it, n):
    '''iterator -> iterator of iterators with groups of size n'''
    return zip(*[iter(it)]*n)
