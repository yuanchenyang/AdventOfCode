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

def grouped(it, n):
    '''iterator -> iterator of iterators with groups of size n'''
    return zip(*[iter(it)]*n)
