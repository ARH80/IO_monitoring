with open('test.txt', 'a') as f:
    string = ""
    for i in range(100000000):
        f.write(F'{i}. Ahmad is here.\n')

with open('test.txt', 'r') as f:
    string = ""
    for i in range(100000000):
        f.readline()

import os
os.remove('test.txt')

