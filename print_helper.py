import sys


def myprint(string = 'This is error output'):
    return print(string, file=sys.stderr)