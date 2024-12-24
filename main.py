import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad


def orthogonality_test(ort_system, deg, eps):
    for i in range (deg):
        result, _ = quad(lambda x: ort_system[i](x) * ort_system[i](x), -1, 1) #нормированность
        if abs(result - 1.) > eps:
            return False
        for j in range (i+1, deg + 1):
            result, _ = quad(lambda x: ort_system[i](x) * ort_system[j](x), -1, 1) #ортогональность
            if abs(result) > eps:
                return False
        return True

#с.зн. А - корни L[n] полинома
def eigenvalues_test(L, eig_values, eps):
    for a in eig_values:
        if (abs(L(a)) > eps):
            return False
    return True

#скалярное произведение (x^i, x^j) на (0, 1) с весом 1
def gram_dot(x):
    if (x % 2) == 1:
        return 0
    else:
        return 2 / (x + 1)

def Gram_Schmidt_method(deg):
#матрица Грама для системы {1, x, ..., x^n}
    G = np.zeros((deg+1, deg+1))
    for i in range(deg+1):
        for j in range(deg+1):
            G[i, j] = gram_dot(i + j)
#Разложение Холецкого с верхнетреугольной матрией G = (U*)U, искомая матрица: U^(-1)
    U = np.linalg.inv(np.linalg.cholesky(G, upper=True))
    ort_system = []
    for j in range(deg + 1):
        ort_system.append(Polynomial(U[:, j]))

    return ort_system


def recurrence_method(deg):
#найдем L0, L1
    ort_system = []
    ort_system.append(Polynomial([1/2 ** 0.5])) #L0 нормирован с весом 1 на (-1;1)
#xL[0] = a[0]L[0] + b[0]L[1], найдем a[0], b[0]
    t1, _ = quad(lambda x: ort_system[0](x) * ort_system[0](x) * x, -1, 1)
    t2, _ = quad(lambda x: ort_system[0](x) * ort_system[0](x), -1, 1)
    alpha = t1 / t2
    alpha_vec = [alpha]
    p = Polynomial([0., 1.]) * ort_system[0] - alpha * ort_system[0]
    t, _ = quad(lambda x: p(x) * p(x), -1, 1)
    beta = t ** 0.5 #нормируем
    beta_vec = [beta]
    ort_system.append(p/beta)

#xL[i] = b[i-1]L[i-1] + a[i]L[i] + b[i]L[i+1]
    for i in range(1, deg):
        t1, _ = quad(lambda x: ort_system[i](x) * ort_system[i](x) * x, -1, 1)
        t2, _ = quad(lambda x: ort_system[i](x) * ort_system[i](x), -1, 1)
        alpha = t1 / t2
        alpha_vec.append(alpha)
        p = Polynomial([0., 1.]) * ort_system[i] - alpha * ort_system[i] - beta_vec[i-1] * ort_system[i-1]
        t, _ = quad(lambda x: p(x) * p(x), -1, 1)
        beta = t ** 0.5
        beta_vec.append(beta)
        ort_system.append(p / beta)

    A = np.zeros((deg, deg))
    for i in range (deg):
        A[i, i] = alpha_vec[i]
        if i > 0:
            A[i, i - 1] = beta_vec[i - 1]
        if i < deg - 1:
            A[i, i + 1] = beta_vec[i]
    eig_values, _ = np.linalg.eig(A)
    return ort_system, eig_values

#Тесты
eps = 10 ** (-5)
deg_P = 4

orthogonal_system = Gram_Schmidt_method(deg_P)
assert(orthogonality_test(orthogonal_system, deg_P, eps))
print("Первый тест пройден")

orthogonal_system, eig_values = recurrence_method(deg_P)
assert(orthogonality_test(orthogonal_system, deg_P, eps))
print("Второй тест пройден")

assert(eigenvalues_test(orthogonal_system[deg_P], eig_values, eps))
print("Третий тест пройден")