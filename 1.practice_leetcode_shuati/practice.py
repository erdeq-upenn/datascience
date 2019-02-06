import os
import sys

f = open('list', 'w')

def lfind(filemane):
    lines = read(filename)
    for line in lines:
        print line[::-1]

filename = sys.argv[1]
lfind(filename)
