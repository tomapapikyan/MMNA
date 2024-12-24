import numpy as np
from numpy.polynomial import chebyshev
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

def remez_method(n, func, eps):
# шаг 1- выбрать стартовые точки (корни T[n+2](x))
    T_coeff = np.zeros(n+3)
    T_coeff[n+2] = 1
    points = chebyshev.Chebyshev(T_coeff).roots()

#шаг 2 - составить и решить СЛАУ, найти p[n], e
    W = np.zeros((n+2, n+2))
    f = np.zeros(n+2)
    while 1:
        str = 0
        for x in points:
            for i in range (n+1):
                W[str][i] = x ** i
            W[str][n + 1] = (-1) ** str
            f[str] = func(x)
            str += 1

        sol = np.linalg.solve(W, f)
        P = Polynomial(sol[0 : n+1])
        e = sol[n+1]

#шаг 3 - найти x* : f(x*) - p(x*) max на [-1, 1]
        Q = func - P
        Q_extremums = Q.deriv().roots()
        if abs(Q(-1)) > abs(Q(1)):
            argmax = -1
            max = abs(Q(-1))
        else:
            argmax = 1
            max = abs(Q(1))
        for x in Q_extremums:
            if (abs(x) <= 1) and (abs(Q(x)) > max):
                max = abs(Q(x))
                argmax = x

#проверить условия останова
        if abs(max - abs(e)) < eps:
            return P

#шаг 3 - замена точки
        if argmax < points[0]:
            if Q(argmax)*Q(points[0]) > 0:
                points[0] = argmax
            else:
                points = np.delete(points, -1)
                points = np.insert(points, 0, argmax)
        elif argmax > points[n+1]:
            if Q(argmax)*Q(points[n+1]) > 0:
                points[n+1] = argmax
            else:
                points = np.delete(points, 0)
                points = np.append(points, argmax)
        else:
            j = 0
            while points[j] < argmax:
                j += 1
            if Q(argmax) * Q(points[j - 1]) > 0:
                points[j - 1] = argmax
            else:
                points[j] = argmax


# test 1
n = 5
eps = 10e-5
p = [0, -1, 0, 0, 1, 1]
pol = Polynomial(p)
res = remez_method(n - 2, pol, eps)


x = np.linspace(-1, 1, 50)
plt.plot(x, pol(x), label = 'исходный')
plt.plot(x, res(x), label = 'интерполянт')
plt.legend()
plt.grid()
plt.show()
