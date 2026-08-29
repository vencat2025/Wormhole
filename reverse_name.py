#!/usr/bin/env python3
import sys

def reverse_name(name: str) -> str:
    return name[::-1]


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        try:
            name = input('Enter name: ')
        except EOFError:
            name = ''
    print(reverse_name(name))


if __name__ == '__main__':
    main()
