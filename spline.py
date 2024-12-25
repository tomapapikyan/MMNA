import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

def Spline_Interpolation(h_vec, f):
#сформируем и решим T*u = p
    n = len(h_vec)-1
    T = np.zeros((n, n))
    p = np.zeros(n)
    x = -1
    for i in range(n):
        T[i][i] = 2*(h_vec[i]+h_vec[i+1])
        if i < n-1:
            T[i][i+1] = h_vec[i+1]
        if i > 0:
            T[i][i-1] = h_vec[i]
        delta0 = (f(x + h_vec[i]) - f(x)) / h_vec[i]
        delta1 = (f(x + h_vec[i] + h_vec[i+1]) - f(x + h_vec[i])) / h_vec[i+1]
        p[i] = 6*(delta1-delta0)
        x += h_vec[i]

    u = np.linalg.solve(T, p)
    u = np.append(u, 0)
    u = np.insert(u, 0, 0) #u0, un = 0
#составим сплайны на каждом отрезке
    spline_list = np.zeros(n+1, dtype=Polynomial)
    x = -1
    for i in range(n+1):
        delta = (f(x + h_vec[i]) - f(x))/ h_vec[i]
        dS = delta - h_vec[i] * u[i] / 3 - h_vec[i] * u[i+1] / 6
        spline_list[i] = Polynomial([f(x), dS * h_vec[i], u[i] * h_vec[i] * h_vec[i] / 2, (u[i + 1] - u[i]) * h_vec[i] * h_vec[i] / 6])
        x += h_vec[i]

    return spline_list

def f_abs(x):
    return abs(x)

def spline(spline_list, h_vec, x):
    t = -1
    for i in range(len(h_vec)):
        if t <= x and x <= t + h_vec[i]:
            return spline_list[i]((x - t) / h_vec[i])
        t += h_vec[i]

N = 3
h = [2 / N for i in range(N)] #равномерная сетка на [-1, 1]
spline_appr = Spline_Interpolation(h, f_abs)

num = 50
x = np.linspace(-1, 1, num)
y = np.zeros_like(x)
for i in range (num):
    y[i] = spline(spline_appr, h, x[i])
plt.plot(x, f_abs(x), label = 'f(x) = |х|')
plt.plot(x, y, label = 'сплайн')
plt.legend()
plt.grid()
plt.show()
