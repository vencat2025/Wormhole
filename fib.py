#!/usr/bin/env python3
import sys

def fibonacci_series(n):
    a, b = 0, 1
    series = []
    for _ in range(n):
        series.append(a)
        a, b = b, a + b
    return series

if __name__ == '__main__':
    # Determine n: from command line argument if provided, else default to 10
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print('Please provide an integer for n.')
            sys.exit(1)
    else:
        n = 10
    print('Fibonacci series (first', n, 'numbers):')
    print(' '.join(map(str, fibonacci_series(n))))
