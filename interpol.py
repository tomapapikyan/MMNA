import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial import chebyshev
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
import matplotlib.pyplot as plt


def Vandermonde(n, func):
    #T_n = np.zeros(n + 1)
    #T_n[n] = 1
    #points = chebyshev.Chebyshev(T_n).roots() #чебышевская сетка
    points = np.linspace(-1, 1, n+1, dtype=np.float64) #равномерная сетка
    W = np.vander(points, increasing=True)
    f = np.array(func(points))
    sol = np.linalg.solve(W, f)
    P = Polynomial(sol)
    return P


def Lagrange(n, func):
    x = np.linspace(-1, 1, n + 1, dtype=np.float64)
    lagrange_basis = [Polynomial([1]) for i in range (n+1)]
    for i in range(n+1):
        for j in range(n+1):
            if j != i:
                lagrange_basis[i] = (lagrange_basis[i]*
                                     Polynomial([-x[j] / (x[i] - x[j]), 1 / (x[i] - x[j])]))
    Ln = Polynomial([0])
    for i in range(n+1):
        Ln = np.polyadd(Ln, lagrange_basis[i]*func(x[i]))[0]
    return Ln


def dot_product(f1, f2):
    res, _ = quad(lambda x: f1(x) * f2(x), -1, 1)
    return res


def Orthogonal(n, func):
#будем раскладывать по базису Лежандра
    basis = [legendre.Legendre.basis(i) for i in range(n + 1)]
    coeff = []
    for i in range(n+1):
        coeff.append(dot_product(basis[i], func) / dot_product(basis[i], basis[i]))
    P = sum([coeff * basis for coeff, basis in zip(coeff, basis)])
    return P

def abs_f(x):
    return abs(x)

n = 4

res = Vandermonde(n, abs_f)
res2 = Orthogonal(n, abs_f)
res3 = Lagrange(n, abs_f)

x = np.linspace(-1, 1, 50)
plt.plot(x, abs_f(x), label = '|x|')
plt.plot(x, res(x), label = 'вандермонд')
plt.plot(x, res2(x), label = 'ортогональные')
plt.plot(x, res3(x), label = 'лагранж')

plt.legend()
plt.grid()
plt.show()
