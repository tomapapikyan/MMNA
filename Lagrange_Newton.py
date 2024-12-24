import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


def divided_difference(k, x, f):
#по лемме о представлении разделенной разности
    res = 0
    for i in range(k+1):
        t = f[i]
        for j in range(k+1):
            if i != j:
                t = t/(x[i] - x[j])
        res += t
    return res


def Lagrange(n, func):
    x = np.linspace(-1, 1, n+1)
    f = np.array(func(x))
    Ln = Polynomial([0])
    for i in range(n+1):
        p = Polynomial([divided_difference(i, x, f)])
        for j in range(i):
            p = np.polymul(p, Polynomial([-x[j], 1]))
        Ln = np.polyadd(Ln, p)[0]
    return Ln

def f(x):
    return x**8

res = Lagrange(5,f)

x = np.linspace(-1, 1, 50)
plt.plot(x, f(x), label = 'f(x)')
plt.plot(x, res(x), label = 'лагранж')
plt.legend()
plt.grid()
plt.show()
