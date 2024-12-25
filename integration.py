import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial import legendre as leg
import matplotlib.pyplot as plt
from scipy.integrate import quad

"""
Считаем для функции веса = 1
"""

def coeff(i, n, t):
    L = Polynomial([1])
    for j in range(n):
        if j != i:
            L = L *Polynomial([-t[j] / (t[i] - t[j]), 1 / (t[i] - t[j])])
    res, _ = quad(lambda x: L(x), -1, 1)
    return res

def Newton_Cotes(n, a, b, f):
    t_points = np.linspace(-1, 1, n)
    x_points = (a+b)/2 + (b-a)/2*t_points

    S = 0
    for i in range(n):
        di = coeff(i, n, t_points)
        S += di*f(x_points[i])
    S *= (b-a)/2
    return S


def Gauss(n, a, b, f):
#узлы - корни соотв ортогонального полинома (с весом 1- лежандра)
    z = np.zeros(n+1)
    z[n] = 1
    x_nods =leg.legroots(z)
    x_points = (a+b)/2 + (b-a)/2*x_nods
#веса находятся как (li, lj) - ск пр элементарных полиномов лагранжа
    weights = np.zeros_like(x_nods)
    for i in range(n):
        l = Polynomial([1])
        for j in range(n):
            if j != i:
                l = l * Polynomial([-x_nods[j] / (x_nods[i] - x_nods[j]), 1 / (x_nods[i] - x_nods[j])])
        weights[i], _ = quad(lambda x: l(x)*l(x), -1, 1)
    S = 0
    for i in range(n):
        S += weights[i] * f(x_points[i])
    S *= (b-a)/2
    return S


def Clenshaw_Curtis(m, a, b, f):
#узлы кленшоу кертиса
    t_points = np.array([np.cos(np.pi * j / m) for j in range (m+1)])
    x_points = (a + b) / 2 + (b - a) / 2 * t_points
    N = m // 2
    S = 0
#cчитаем веса
    for j in range(m + 1):
        w = 1 / 2
        base = 2 * j * np.pi / m
        arg = 0
        for k in range(1, N):
            arg += base
            w += np.cos(arg) / (1 - 4 * k ** 2)
        if 2*N == m:
            w += 0.5 * np.cos(j * np.pi)/(1-m*m)
        else:
            w += np.cos(arg+base)/(1-m*m)
        w *= 4 / m

        if j == 0 or j == m:
            w /= 2

        S += w * f(x_points[j])

    S *= (b - a) / 2
    return S


#test
def f(x):
    return np.cos(x)
def F(x):
    return np.sin(x)

a = -2
b = 1

real_int = F(b)-F(a)

n = np.array([i for i in range(10)])

Newton_Cotes_err = np.zeros(10)
Gauss_err = np.zeros(10)
Clenshaw_Curtis_err = np.zeros(10)

for i in range(10):
    Newton_Cotes_err[i] = abs(real_int - Newton_Cotes(n[i], a, b, f))
    Gauss_err[i] = abs(real_int - Gauss(n[i], a, b, f))
    Clenshaw_Curtis_err[i] = abs(real_int - Clenshaw_Curtis(n[i], a, b, f))


plt.plot(n, Newton_Cotes_err, label = 'Newton-Cotes err')
plt.plot(n, Gauss_err, label='Gauss err')
plt.plot(n, Clenshaw_Curtis_err, label='Clenshaw_Curtis err')
plt.legend()
plt.grid()
plt.show()
