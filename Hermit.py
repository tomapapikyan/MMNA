import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import math


def divided_difference(start, k, x, f_dict):
#k-порядок
    if k == 0:
        return f_dict[x[start]][0]
    if x[start] == x[start+k]:
        return f_dict[x[start]][k]/math.factorial(k)
    else:
        t2 = divided_difference(start+1, k-1, x, f_dict)
        t1 = divided_difference(start, k-1, x, f_dict)
        return (t2-t1)/(x[start+k]-x[start])


#(x = [x0, ..., xm], f_dict= [x0 : [f, df, d2f, ...], ..., xm: [f, df, ...])
def Hermit(n, x, f_dict):
    Hn = Polynomial([0])
    for i in range(n+1):
        p = Polynomial([divided_difference(0, i, x, f_dict)])
        #print(p)
        for j in range(i):
            p = np.polymul(p, Polynomial([-x[j], 1]))
        Hn = np.polyadd(Hn, p)[0]
        print (Hn)
    return Hn

"""
#test1 без производных
def f(x):
    return x**5

deg = 5 #проверяется точное совпадение
x = np.linspace(-1, 1, deg+1)
f_dict = {}
for p in x:
    f_dict[p] = np.array([f(p)])
print(f_dict)
res = Hermit(deg, x,f_dict)
"""

#test2 с производными
def f(x):
    return x**7+1

deg = 5 #приблизим полиномом меньшей степени
x = np.array([-1, -1, 0, 0, 1, 1])
f_dict = {}
f_dict[-1] = np.array([0, 7])
f_dict[0] = np.array([1, 0])
f_dict[1] = np.array([2, 7])

res = Hermit(deg, x,f_dict)


x = np.linspace(-1, 1, 50)
plt.plot(x, f(x), label = 'f(x) = x^7+1')
plt.plot(x, res(x), label = 'эрмит 5 степени')
plt.legend()
plt.grid()
plt.show()
