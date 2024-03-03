import math


def f(x):
    return 127/4*x**2 - 61/4*x + 2

def f_min(a,b,l):
    left = a
    right = b
    k = 0
    y = 0
    while right - left > l:
        y = left + (3-math.sqrt(5))/2*(right-left)
        z = left + right - y
        if f(y) < f(z):
            right = z
            y -= left + right
        else:
            left = 