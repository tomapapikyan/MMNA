import numpy as np
from scipy.stats import qmc


def MonteCarlo(f, l1, l2, n):
    x = np.random.uniform(l1[0], l1[1], n)
    y = np.random.uniform(l2[0], l2[1], n)
    sum = 0
    for i in range(n):
        sum += f(x[i], y[i])
    val = (l1[1]-l1[0])*(l2[1]-l2[0])

    return val/n*sum


def RQMC(f, l1, l2, n, s):
    sampler = qmc.Sobol(d=2, scramble=s)
    sample = sampler.random_base2(n)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = l1[0] + (l2[1]-l1[0]) * sample[i][0]
        y[i] = l2[0] + (l2[1]-l1[0]) * sample[i][1]
    sum = 0
    for i in range(n):
        sum += f(x[i], y[i])
    val = (l1[1] - l1[0]) * (l2[1] - l2[0])

    return val / n * sum


def H(x, y): #пример для вычисления pi
    if x**2 + y**2 <= 1:
        return 1
    else:
        return 0

print(MonteCarlo(H, [-1, 1], [-1, 1], 1000)) #обычный
print(RQMC(H, [-1, 1], [-1, 1], 20, False)) #квази-случайный
print(RQMC(H, [-1, 1], [-1, 1], 20, True)) #рандомизированный квази-случайный