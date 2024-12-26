import numpy as np

def Newton(f, df, x0, eps):
    x = x0
    while(abs(f(x))>eps):
        x = x - f(x)/df(x)
    return x

def f(x):
    return np.cos(x)-x**3

def df(x):
    return -np.sin(x)-3*x**2

x0 = 1/2
eps = 10 ** (-5)
print(Newton(f, df, x0, 10**(-5)))