import sys

while True:
    key = sys.stdin.read(1)
    print('u')
    if key == '\x1b':
        break
