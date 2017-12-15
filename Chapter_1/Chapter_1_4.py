# coding = utf-8
# python basic grammar
t = (1, 'abc', 0.4)
# t[0] = 2  TypeError: 'tuple' object does not support item assignment

l = [1, 'abc', 0.4]
l[0] = 2
l[0] += 1
l[0]

l[0] -= 2
l[0]

True and True

True and False

True or False

False or False

not True

l = [1, 'abc', 0.4]
t = (1, 'abc', 0.4)
d = {1: '1', 'abc': 0.1, 0.4: 80}

0.4 in l

1 in t

'abc' in d

0.1 in d

b = True

if b:
    print "It's True!"
else:
    print "It's False!"

b = False
c = True

if b:
    print "b is True!"
elif c:
    print "c is True!"
else:
    print "Both are False!"

b = False
c = False
if b:
    print "b is True!"
elif c:
    print "c is True!"
else:
    print "Both are False!"

d = {1: '1', 'abc': 0.1, 0.4: 80}
for k in d:
    print k, ":", d[k]


def foo(x):
    return x ** 2


foo(8.0)

import math

math.exp(2)

from math import exp

exp(2)

from math import exp as ep

ep(2)
