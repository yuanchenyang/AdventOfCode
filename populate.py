import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--day', '-d', type=int)
args = parser.parse_args()
# parser.add_argument('--session_token', '-t', type=str)

TEST_INPUT_TEMPLATE = """
day_{d}_test_input = ''''''
"""

SOL_TEMPLATE = """
def day_{d}_common(s):
    return

def day_{d}a(s):
    '''
    >>> day_{d}a(day_{d}_test_input)
    '''
    return day_{d}_common(s)

def day_{d}b(s):
    '''
    >>> day_{d}b(day_{d}_test_input)
    '''
    return day_{d}_common(s)
"""

with open(f'inputs/{args.day}.in', 'w') as f:
    f.write('')

with open(f'test_inputs.py', 'a') as f:
    f.write(TEST_INPUT_TEMPLATE.format(d=args.day))

with open(f'sols.py', 'a') as f:
    f.write(SOL_TEMPLATE.format(d=args.day))
